import argparse
import sys
import os
import numpy as np
import nrrd

from .utilities.path_utils import metadata
from .points import id_to_points, gene_to_points, write_values_to_meshview
from .volume import id_to_volume, gene_to_volume, interpolate, write_nifti

# config constants
image_folder_path = "/mnt/g/AllenDataalignmentProj/resolutionPixelSizeMetadata/ISH/"
path_to_registration_files = "/mnt/g/Allen_Realignment_EBRAINS_dataset/registration_data"


def main():
    parser = argparse.ArgumentParser(description="Allen Quantifier CLI")
    subs = parser.add_subparsers(dest="cmd")

    # points
    p1 = subs.add_parser("points", help="point cloud")
    p1.add_argument("--id", type=int)
    p1.add_argument("--gene", type=str)
    p1.add_argument("--mode", choices=["expression", "histology"], default="expression")
    p1.add_argument("--res", type=int, default=25)
    p1.add_argument("--cut", type=int, default=0)

    # unify volume + process-gene
    vol_p = subs.add_parser("volume", help="compute or save volume (ID or gene)")
    vol_p.add_argument("--id", type=int, help="Experiment ID")
    vol_p.add_argument("--gene", type=str, help="Gene name")
    vol_p.add_argument(
        "--mode", choices=["expression", "histology"], default="expression"
    )
    vol_p.add_argument("--res", type=int, default=10, help="resolution in µm")
    vol_p.add_argument("--cut", type=int, default=0)  # if needed for points
    vol_p.add_argument(
        "--interpolate",
        default=True,
        action="store_true",
        help="fill missing voxels via interpolation",
    )
    vol_p.add_argument("--k", type=int, default=5, help="neighbors for interpolation")
    vol_p.add_argument("--out-nifti", help="path to save as NIfTI (uint8)")

    args = parser.parse_args()

    if args.cmd == "points":
        if args.id:
            v, p = id_to_points(
                args.id,
                image_folder_path,
                path_to_registration_files,
                resolution=args.res,
                cutoff=args.cut,
                mode=args.mode,
            )
            write_values_to_meshview(v, p, f"{args.id}_{args.mode}_cut{args.cut}.json")
        elif args.gene:
            v, p = gene_to_points(
                args.gene,
                image_folder_path,
                path_to_registration_files,
                cutoff=args.cut,
                mode=args.mode,
                resolution=args.res,
            )
            write_values_to_meshview(
                v, p, f"{args.gene}_{args.mode}_cut{args.cut}.json"
            )
        else:
            parser.error("points: require --id or --gene")

    elif args.cmd == "volume":
        if not (args.id or args.gene):
            parser.error("volume: require --id or --gene")

        # select core volume function
        if args.gene:
            vol, freq = gene_to_volume(
                args.gene,
                image_folder_path,
                path_to_registration_files,
                mode=args.mode,
                resolution=args.res,
                return_frequencies=True,
                do_interpolation=args.interpolate,
                missing_fill=0,
            )
        else:
            vol, freq = id_to_volume(
                args.id,
                image_folder_path,
                path_to_registration_files,
                mode=args.mode,
                resolution=args.res,
                return_frequencies=True,
                do_interpolation=args.interpolate,
                missing_fill=0,
            )
        # output
        write_nifti(vol, resolution=args.res, output_path=args.out_nifti)

    else:
        parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
