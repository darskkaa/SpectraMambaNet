import argparse
import sys
import time

import torch

from model.net import SpectraMambaNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=352, help="Input resolution (matches train/eval res)")
    parser.add_argument("--fps-batch", type=int, default=1, help="Batch size for FPS latency measurement")
    parser.add_argument("--mem-batch", type=int, default=8, help="Batch size for peak-memory measurement (matches train.py's per-device batch)")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--use-boundary-attn", action="store_true",
                         help="Match a run trained with --use-boundary-attn (adds the edge-attention module)")
    parser.add_argument("--use-wavelet-decoder", action="store_true",
                         help="Match a run trained with --use-wavelet-decoder (adds the wavelet decoder branch)")
    # These mirror train.py's defaults on purpose. Before this, the script left
    # use_fcc_decoder at the constructor's False while train.py defaults it to True, so every
    # params/FLOPs/FPS/memory number it printed described the legacy decoder path — a
    # different network from the one in the accuracy table.
    parser.add_argument("--no-fcc-decoder", action="store_false", dest="use_fcc_decoder", default=True,
                         help="Profile the legacy decoder chain instead of the FCC decoder (matches --no-fcc-decoder at train time)")
    parser.add_argument("--fcc-channel", type=int, default=64)
    parser.add_argument("--fcc-reduce-octave", action="store_true",
                        help="Match --fcc-reduce-octave at train time. It is the single largest "
                             "parameter lever we have (~-7.0 M / ~-1.4 GMACs), so without it "
                             "this script cannot profile the config S-max actually trains.")
    parser.add_argument("--no-fcc-region-calib", action="store_false", dest="fcc_region_calib", default=True)
    parser.add_argument("--fcc-region-proto", action="store_true",
                        help="Match --fcc-region-proto at train time. It extends BoundaryRegionCalib "
                             "with a prototype head, so without it this script reports a parameter "
                             "count the shipped config does not have.")
    parser.add_argument("--use-rfp", action="store_true",
                        help="Profile with the Rank-normalized Frequency Prior (matches --use-rfp at train time)")
    parser.add_argument("--mamba-bidirectional", action="store_true")
    parser.add_argument("--mamba-stage", type=str, default='bottleneck',
                        choices=['bottleneck', 'h8', 'both', 'h4', 'h4h8'],
                        help="h8 moves the scan from 11-token to 44-token axes; h4 to 88-token "
                             "ones on f0, and h4h8 does both. Each has FEWER parameters and "
                             "MORE compute than the bottleneck, so params alone will mislead - "
                             "this is the number that has to be measured. thop does not trace "
                             "the selective scan, so the printed MACs UNDER-count every one of "
                             "these; the gap grows with token count.")
    parser.add_argument("--mamba-scan", type=str, default='axial', choices=['axial', 'raster'],
                        help="Match --mamba-scan at train time. Parameter count is identical "
                             "either way and thop traces neither, so this moves no number here "
                             "— it is present so the [config] line this script prints is the "
                             "same string as the run's, which is what the cost/Dice pairing "
                             "is checked against.")
    parser.add_argument("--encoder", type=str, default='pvt_v2_b2',
                        choices=['pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5'],
                        help="Match --encoder at train time. B2-B5 keep the same decoder "
                             "channels, so this moves params/FLOPs and nothing else.")
    parser.add_argument("--edge-stride", type=int, default=8, choices=[4, 8])
    parser.add_argument("--no-mamba-bottleneck", action="store_false",
                        dest="use_mamba_bottleneck", default=True)
    parser.add_argument("--no-freq-axis", action="store_false", dest="mamba_freq_axis",
                        default=True,
                        help="Match --no-freq-axis at train time (ablation A1). Drops norm_f/"
                             "mamba_f/mag_head/spectral_fuse from every tri-axial block. A1's "
                             "claim is that it is NEAR-PARAMETER-MATCHED to the full model, so "
                             "that gap has to be measured with this script rather than asserted.")
    # Accepted but inert, for the same reason --mamba-scan is: the ablation cell profiles each arm
    # by appending that arm's TRAINING flags verbatim, so a flag this parser does not know makes
    # argparse exit 2 and the arm loses its params/FLOPs row. Neither of these moves a number --
    # freq_gain only bounds an activation and freq_gain_scale only sets the bound -- but both are
    # printed in the [config] line below, which is the string the cost/Dice pairing is checked on.
    parser.add_argument("--freq-gain", type=str, default='clamp', choices=['clamp', 'tanh'])
    parser.add_argument("--freq-gain-scale", type=float, default=3.0)
    parser.add_argument("--freq-gain-shared", action="store_true",
                        help="Match --freq-gain-shared at train time. mag_head becomes "
                             "Linear(dim, 1) instead of Linear(dim, dim), which is the only "
                             "one of the --freq-gain* flags that moves the parameter count — "
                             "and it moves it down, so the constrained arm is strictly cheaper "
                             "than A0 as well as strictly less free.")
    parser.add_argument("--third-axis", type=str, default='freq', choices=['freq', 'raster'],
                        help="Match --third-axis at train time. The 'raster' control builds the "
                             "identical modules, so this must profile to the SAME parameter "
                             "count as the 'freq' arm — if it does not, the control is not "
                             "params-matched and its row is worthless.")
    args = parser.parse_args()

    if args.use_fcc_decoder and (args.use_boundary_attn or args.use_wavelet_decoder):
        print("FATAL: --use-boundary-attn/--use-wavelet-decoder are no-ops under the FCC decoder")
        print("(SpectraMambaNet.__init__ returns before constructing them). Pass --no-fcc-decoder")
        print("to profile that path, or drop the flag — train.py refuses the same combination.")
        sys.exit(1)

    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        print("FATAL: mamba_ssm is not installed. model.net silently falls back to a Linear+GELU")
        print("stub without it — profiling that stub would report meaningless params/FLOPs/latency")
        print("numbers instead of the real Mamba SSM's. Install mamba_ssm (pip install mamba-ssm)")
        print("and run this on a CUDA machine before profiling.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("FATAL: mamba_ssm's selective-scan kernel requires CUDA; this repo's Mamba blocks")
        print("won't run on CPU. Run this script on a CUDA machine (e.g. the Colab GPU runtime).")
        sys.exit(1)

    try:
        from thop import profile as thop_profile
    except ImportError:
        print("FATAL: thop is not installed. pip install thop.")
        sys.exit(1)

    device = torch.device("cuda")
    model = SpectraMambaNet(
        pretrained_encoder=False,
        use_boundary_attn=args.use_boundary_attn,
        use_wavelet_decoder=args.use_wavelet_decoder,
        use_fcc_decoder=args.use_fcc_decoder,
        fcc_channel=args.fcc_channel,
        fcc_reduce_octave=args.fcc_reduce_octave,
        fcc_region_calib=args.fcc_region_calib,
        fcc_region_proto=args.fcc_region_proto,
        use_rfp=args.use_rfp,
        use_mamba_bottleneck=args.use_mamba_bottleneck,
        mamba_bidirectional=args.mamba_bidirectional,
        mamba_stage=args.mamba_stage,
        mamba_scan=args.mamba_scan,
        mamba_freq_axis=args.mamba_freq_axis,
        freq_gain=args.freq_gain,
        freq_gain_scale=args.freq_gain_scale,
        freq_gain_shared=args.freq_gain_shared,
        third_axis=args.third_axis,
        edge_stride=args.edge_stride,
        encoder_name=args.encoder,
    ).to(device)
    print(f"[config] encoder={args.encoder} "
          f"use_fcc_decoder={args.use_fcc_decoder} fcc_channel={args.fcc_channel} "
          f"fcc_reduce_octave={args.fcc_reduce_octave} "
          f"fcc_region_calib={args.fcc_region_calib} fcc_region_proto={args.fcc_region_proto} "
          f"use_boundary_attn={args.use_boundary_attn} "
          f"use_wavelet_decoder={args.use_wavelet_decoder} use_rfp={args.use_rfp} "
          f"use_mamba_bottleneck={args.use_mamba_bottleneck} "
          f"mamba_bidirectional={args.mamba_bidirectional} mamba_stage={args.mamba_stage} "
          f"mamba_scan={args.mamba_scan} mamba_freq_axis={args.mamba_freq_axis} "
          f"freq_gain={args.freq_gain} freq_gain_scale={args.freq_gain_scale} "
          f"freq_gain_shared={args.freq_gain_shared} third_axis={args.third_axis} "
          f"edge_stride={args.edge_stride} "
          f"— pass these flags to match the checkpoint you're profiling.")

    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    model.eval()
    dummy = torch.randn(1, 3, args.size, args.size, device=device)
    macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
    # 2x MACs = FLOPs, matching BAMPolyp's (bampolyp.py) convention.
    flops_g = 2 * macs / 1e9

    # thop is the number, because thop is what the field publishes. Measured on this harness at
    # 352x352: CTNet profiles at 15.139 against its published 15.14, PraNet at 13.150 against
    # 13.1, Polyp-PVT at 10.018 against 10.1. Every competitor score a reader will compare us
    # against came out of this tool, so ours has to as well or the column is not a column.
    #
    # thop hooks nn.Module.forward, and Mamba.forward calls none of its submodules, so it counts
    # zero for a Mamba block. That is a property of the convention, not of our model, and Mamba
    # papers publish under it unchanged -- BAMPolyp (ICCVW 2025) reports 3.13 G bare. Do not add
    # a hand-derived scan term and do not swap in a dispatch-level counter for our row alone:
    # dispatch also finds ~8% more in everyone else's attention, so a mixed column charges us
    # twice. If the tool ever changes, it changes for every row at once.

    # FPS: batch-1 forward-only latency (BAMPolyp convention), at native training/eval resolution.
    fps_input = torch.randn(args.fps_batch, 3, args.size, args.size, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(fps_input)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.repeat):
            model(fps_input)
        torch.cuda.synchronize()
    elapsed = time.time() - start
    fps = (args.repeat * args.fps_batch) / elapsed

    # Peak GPU memory: forward+backward at train.py's actual per-device batch (8), so this
    # reflects our real training footprint rather than an arbitrary benchmark batch size.
    # NOTE: this is single-scale, no AMP, no optimizer state — train.py's real training_step
    # runs 3 multi-scale passes under AMP with AdamW, all resident before one backward call.
    # Treat this number as directional only; confirm with a real train.py smoke run before
    # raising --mem-batch further.
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    mem_input = torch.randn(args.mem_batch, 3, args.size, args.size, device=device)
    outputs = model(mem_input)
    loss = sum(o.float().abs().mean() for o in outputs)
    loss.backward()
    torch.cuda.synchronize()
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print(f"Params (M):   {params_m:.2f}")
    print(f"GMACs:        {macs / 1e9:.3f}  [thop, input {args.size}x{args.size}, batch=1 — "
          f"THIS IS THE PAPER NUMBER, same tool every competitor row used]")
    print(f"FLOPs (G):    {flops_g:.2f}  [2 x MACs; do not mix the two conventions in one column]")
    print(f"FPS:          {fps:.1f}  [batch={args.fps_batch}, input {args.size}x{args.size}, {args.repeat} reps]")
    print(f"Peak GPU mem: {peak_mem_gb:.2f} GB  [fwd+bwd, batch={args.mem_batch}]")


if __name__ == "__main__":
    main()
