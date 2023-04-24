class Factor:
    """
    This the class defines a multidimensional conditional probability.
    """
    def __init__(self, table=None, table_len=None, variables=None, pdfs=None, equivalent_variables=None, na_values=None):
        self.table = table
        self.table_len = table_len
        self.variables = variables
        self.equivalent_variables = equivalent_variables
        self.pdfs = pdfs
        self.cardinalities = dict()
        for i, var in enumerate(self.variables):
            if type(pdfs) == dict:
                self.cardinalities[var] = len(pdfs[var])
                if equivalent_variables and len(equivalent_variables) != 0:
                    self.cardinalities[equivalent_variables[i]] = pdfs[var]
            else:
                self.cardinalities[var] = pdfs.shape[i]
                if equivalent_variables and len(equivalent_variables) != 0:
                    self.cardinalities[equivalent_variables[i]] = pdfs.shape[i]
        self.na_values = na_values  # the percentage of data, which is not nan, so the variable name is misleading.


class Group_Factor:
    """
        This the class defines a multidimensional conditional probability on a group of tables.
    """
    def __init__(self, tables, tables_size, variables, pdfs, bin_modes, equivalent_groups=None,
                 table_key_equivalent_group=None, na_values=None, join_cond=None):
        self.table = tables
        self.tables_size = tables_size
        self.variables = variables
        self.pdfs = pdfs
        self.bin_modes = bin_modes
        self.equivalent_groups = equivalent_groups
        self.table_key_equivalent_group = table_key_equivalent_group
        self.na_values = na_values
        self.join_cond = join_cond
