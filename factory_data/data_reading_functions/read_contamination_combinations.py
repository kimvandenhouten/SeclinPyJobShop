import pandas as pd
import itertools


def read_contamination_combinations(contamination_path: str, def_enzymes_path: str, translation_table: pd.DataFrame):
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

    # Create all possible combinations based on keys
    keys = translation_table["key"].tolist()
    translation_dict = translation_table.set_index('key').to_dict(orient='index')
    two_element_combinations = list(itertools.combinations(keys, 2))

    # Translate name pairs to number pairs
    number_pairs = [(int(name_to_number[row]), int(name_to_number[col])) for row, col in pairs]
    final_pairs = [pair for pair in two_element_combinations if (translation_dict[pair[0]]['SKU EoF'],
                                                                 translation_dict[pair[1]]['SKU EoF']) in number_pairs]

    return final_pairs


if __name__ == '__main__':
    number_pairs = read_contamination_combinations('factory_data/seclin_recipes/FAM and MF contamination.csv', 'factory_data/seclin_recipes/File_D_DEF_Enzymes.csv')

