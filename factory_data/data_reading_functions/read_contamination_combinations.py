import pandas as pd


def read_contamination_combinations(contamination_path: str, def_enzymes_path: str):
    """
    TODO: can we replace this function by taking it from the original table provided by SN
    TODO: standardized way for using keys / ids / SKU EoFs and so on

    This function preprocesses the contamination matrix that holds for the cleaning times at the FAM/MF level.

    Args:
        contamination_path (str): the path to the contamination matrix.
        def_enzymes_path (str): the path to the enzymes translation.
    """

    # Store sequence-dependent cleaning times for FAM/MF
    df = pd.read_csv(contamination_path, index_col=0, delimiter=";")

    # Get the translation
    translation_df = pd.read_csv(def_enzymes_path, header=None,
                                 names=['id', 'number', 'name'])

    # Create name -> number mapping
    name_to_number = dict(zip(translation_df['name'], translation_df['number']))

    # Get all pairs where the value is 1
    pairs = [(row, col) for row in df.index for col in df.columns if df.loc[row, col] == 1]

    # Translate name pairs to number pairs
    number_pairs = [(name_to_number[row], name_to_number[col]) for row, col in pairs]
    assert len(pairs) == 47

    # TODO: translate number pairs to product ids in factory

    return number_pairs


if __name__ == '__main__':
    number_pairs = read_contamination_combinations('factory_data/seclin_recipes/FAM and MF contamination.csv', 'factory_data/seclin_recipes/File_D_DEF_Enzymes.csv')

