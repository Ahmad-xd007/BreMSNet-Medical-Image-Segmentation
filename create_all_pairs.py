import pandas as pd

from prepare_cbis_pairs import build_cbis_mass_pairs
from prepare_inbreast_masks import generate_inbreast_masks


def main():

    print("=" * 60)
    print("Generating CBIS-DDSM pairs...")
    print("=" * 60)

    cbis_pairs = build_cbis_mass_pairs(
        "/home/tq_ahmad/Dataset/CBIS"
    )

    print("\n")

    print("=" * 60)
    print("Generating INbreast pairs...")
    print("=" * 60)

    inbreast_pairs = generate_inbreast_masks(
        "/home/tq_ahmad/Dataset/INbreast Release 1.0"
    )

    print("\n")

    # ---------------------------------------------------
    # Convert to DataFrames
    # ---------------------------------------------------

    cbis_df = pd.DataFrame(cbis_pairs)
    inbreast_df = pd.DataFrame(inbreast_pairs)

    # ---------------------------------------------------
    # Save individual CSVs
    # ---------------------------------------------------

    cbis_csv = "/home/tq_ahmad/Dataset/cbis_mass_pairs.csv"
    inbreast_csv = "/home/tq_ahmad/Dataset/inbreast_pairs.csv"

    cbis_df.to_csv(
        cbis_csv,
        index=False
    )

    inbreast_df.to_csv(
        inbreast_csv,
        index=False
    )

    # ---------------------------------------------------
    # Create combined dataset
    # ---------------------------------------------------

    all_df = pd.concat(
        [cbis_df, inbreast_df],
        ignore_index=True
    )

    all_csv = "/home/tq_ahmad/Dataset/all_pairs.csv"

    all_df.to_csv(
        all_csv,
        index=False
    )

    # ---------------------------------------------------
    # Print summary
    # ---------------------------------------------------

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"CBIS pairs      : {len(cbis_df)}")
    print(f"INbreast pairs  : {len(inbreast_df)}")
    print(f"Total pairs     : {len(all_df)}")

    print("\nSaved files:")
    print(cbis_csv)
    print(inbreast_csv)
    print(all_csv)

    print("=" * 60)


if __name__ == "__main__":
    main()