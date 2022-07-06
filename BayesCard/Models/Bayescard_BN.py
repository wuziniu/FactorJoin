import numpy as np
import logging
import time
import copy
from BayesCard.Models.BN_single_model import BN_Single
from Pgmpy.factors.discrete.CPD import TabularCPD
from Pgmpy.models import BayesianModel
from Pgmpy.inference import VariableEliminationJIT
logger = logging.getLogger(__name__)


def multi_dim_index(a, index, new_value):
    assert a.ndim == len(index) == new_value.ndim
    new_index = []
    n = len(index)
    for i, ind in enumerate(index):
        ind = np.asarray(ind)
        if i != n-1:
            new_shape = tuple([-1] + [1]*(n-i-1))
        else:
            new_shape = -1
        new_index.append(ind.reshape(new_shape))
    a[tuple(new_index)] = new_value
    return a

class Bayescard_BN(BN_Single):
    """
    Build a single Bayesian Network for a single table using pgmpy
    """

    def __init__(self, table_name, id_attributes, bin_size, column_names=None, nrows=None,
                 meta_types=None, null_values=None,
                 method='Pome', debug=True, infer_algo=None):
        """
        table_name: name of the table
        id_attributes: the id columns in this table
        column_names: the name of the columns
        table_meta_data: the information about the tables
        meta_types: the information about attribute types
        full_join_size: full outer join size of the data this BN is built on
        infer_algo: inference method, choose between 'exact', 'BP'
        """
        BN_Single.__init__(self, table_name, null_values, method, debug)
        self.table_name = table_name
        self.id_attributes = id_attributes
        self.bin_size = bin_size
        self.meta_types = meta_types
        self.null_values = null_values
        self.column_names = column_names
        self.nrows = nrows
        self.infer_algo = infer_algo
        self.infer_machine = None
        self.cpds = None


    def build_from_data(self, dataset, attr_type=None, sample_size=1000000, n_mcv=30, n_bins=60, ignore_cols=[],
                        algorithm="chow-liu", drop_na=True, max_parents=-1, root=0, n_jobs=8, discretized=False):
        """ Build the Pomegranate model from data, including structure learning and paramter learning
            ::Param:: dataset: pandas.dataframe
                      attr_type: type of attributes (binary, discrete or continuous)
                      sample_size: subsample the number of rows to use to learn structure
                      n_mcv: for categorical data we keep the top n most common values and bin the rest
                      n_bins: number of bins for histogram, larger n_bins will provide more accuracy but less efficiency
            for other parameters, pomegranate gives a detailed explaination:
            https://pomegranate.readthedocs.io/en/latest/BayesianNetwork.html
        """
        self.algorithm = algorithm
        if algorithm != "junction":
            discrete_table = self.learn_model_structure(dataset, self.nrows, self.id_attributes,
                                                        attr_type, sample_size,
                                                        n_mcv, n_bins, ignore_cols, algorithm,
                                                        drop_na, max_parents, root, n_jobs,
                                                        return_dataset=True, discretized=discretized)
        else:
            discrete_table = self.learn_model_structure(dataset, self.nrows, self.id_attributes,
                                                        attr_type, sample_size,
                                                        n_mcv, n_bins, ignore_cols, 'chow-liu',
                                                        drop_na, max_parents, root, n_jobs,
                                                        return_dataset=True, discretized=discretized)
        self.data_length = len(discrete_table)
        if self.nrows is None:
            self.nrows = len(discrete_table)
        spec = []
        orphans = []
        for i, parents in enumerate(self.structure):
            for p in parents:
                spec.append((self.node_names[p], self.node_names[i]))
            if not parents:
                orphans.append(self.node_names[i])
        if self.debug:
            logger.info(f"Model spec{spec}")
        self.model = BayesianModel(spec)
        for o in orphans:
            self.model.add_node(o)
        logger.info('calling pgm.BayesianModel.fit...')
        t = time.time()
        self.model.fit(discrete_table)
        if algorithm == "junction":
            try:
                self.model = self.model.to_junction_tree()
            except:
                self.model = self.model
                logger.warning(
                    "This BN is not able to transform into junction tree, probably because "
                    "it's not connected, just use BN")
        logger.info(f"done, took {time.time() - t} secs.")
        print(f"done, parameter learning took {time.time() - t} secs.")
        self.legitimacy_check()
        #self.init_inference_method()

    def realign(self, encode_value, n_distinct):
        """
        Discard the invalid and duplicated values in encode_value and n_distinct and realign the two
        """
        if type(encode_value) != list and type(n_distinct) != list:
            return encode_value, n_distinct

        assert len(encode_value) == len(n_distinct)
        res_value = []
        res_n_distinct = []
        for i, c in enumerate(encode_value):
            if c is not None:
                if c in res_value:
                    index = res_value.index(c)
                    res_n_distinct[index] += n_distinct[i]
                    res_n_distinct[index] = min(res_n_distinct[index], 1)
                else:
                    res_value.append(c)
                    res_n_distinct.append(n_distinct[i])
        return res_value, res_n_distinct

    def update_from_data(self, dataset):
        """
        Preserve the structure and only incrementally update the parameters of BN.
        Currently only implemented data insertion. Data deletion can be done in a similar way.
        """
        t = time.time()
        self.insert_len = len(dataset)
        self.n_in_bin_update = copy.deepcopy(self.n_in_bin)
        self.encoding_update = copy.deepcopy(self.encoding)
        self.mapping_update = copy.deepcopy(self.mapping)

        discrete_table = self.process_update_dataset(dataset)
        print(f"Discretizing table took {time.time() - t} secs.")
        t = time.time()
        incremental_model = copy.deepcopy(self.model)
        incremental_model.fit(discrete_table)

        # incremental parameter updating
        for i, cpd in enumerate(self.model.cpds):
            new_cpd = incremental_model.cpds[i]
            assert set(cpd.state_names.keys()) == set(new_cpd.state_names.keys()), "cpd attribute name mismatch"
            assert cpd.variable == new_cpd.variable, "variable mismatch"
            self.model.cpds[i] = self.update_cpd_table(cpd, new_cpd)

        # changing meta-info
        self.nrows += self.insert_len
        self.mapping = self.mapping_update
        self.encoding = self.encoding_update
        self.n_in_bin = self.n_in_bin_update
        self.legitimacy_check()

        print(f"done, incremental parameter updating took {time.time() - t} secs.")
        self.init_inference_method()

    def update_cpd_table(self, old_cpd, new_cpd):
        """
        Incrementally update the value of one cpd table
        """
        var = old_cpd.variable
        ret_cpd_variable = var
        ret_cpd_evidence = []
        ret_cpd_evidence_card = []
        ret_cpd_state_names = dict()
        ret_values_shape = []
        
        for col in old_cpd.state_names:
            if self.attr_type[col] == "continuous":
                ret_cpd_state_names[col] = list(self.mapping_update[col].keys())
            elif col in self.id_attributes:
                ret_cpd_state_names[col] = list(self.encoding_update[col])
            else:
                ret_cpd_state_names[col] = list(set(self.encoding_update[col].values()))
            if col == var:
                ret_cpd_variable_card = len(ret_cpd_state_names[col])
            else:
                ret_cpd_evidence.append(col)
                ret_cpd_evidence_card.append(len(ret_cpd_state_names[col]))
            ret_values_shape.append(len(ret_cpd_state_names[col]))
        ret_values_old = np.zeros(tuple(ret_values_shape))
        old_index = []
        for col in old_cpd.state_names:
            old_index.append([ret_cpd_state_names[col].index(x) for x in old_cpd.state_names[col]])
        ret_values_old = multi_dim_index(ret_values_old, old_index, old_cpd.values)

        ret_values_new = np.zeros(tuple(ret_values_shape))
        new_index = []
        for col in old_cpd.state_names:
            new_index.append([ret_cpd_state_names[col].index(x) for x in new_cpd.state_names[col]])
        ret_values_new = multi_dim_index(ret_values_new, new_index, new_cpd.values)

        ret_values = self.nrows * ret_values_old + self.insert_len * ret_values_new
        ret_values = ret_values.reshape((ret_values.shape[0], -1))
        ret_cpd = TabularCPD(ret_cpd_variable, ret_cpd_variable_card, ret_values, None, ret_cpd_evidence,
                             ret_cpd_evidence_card, state_names=ret_cpd_state_names)
        ret_cpd.normalize()
        return ret_cpd

    def init_inference_method(self):
        """
        Initial the inference method for query
        """
        assert self.algorithm == "chow-liu", "Currently JIT only supports CLT"
        from Pgmpy.inference import VariableEliminationJIT
        cpds, topological_order, topological_order_node = self.align_cpds_in_topological()
        self.cpds = cpds
        self.topological_order = topological_order
        self.topological_order_node = topological_order_node
        self.infer_algo = "exact-jit"
        self.infer_machine = VariableEliminationJIT(self.model, cpds, topological_order, topological_order_node,
                                                    fanouts=self.id_attributes)


    def continuous_range_map(self, col, range):
        def cal_coverage(l, r, target):
            tl = target.left
            tr = target.right
            if l >= tr: return 0
            if r <= tl: return 0
            if r > tr:
                if l < tl:
                    return 1
                else:
                    return (tr - l) / (tr - tl)
            else:
                if l > tl:
                    return (r - l) / (tr - tl)
                else:
                    return (r - tl) / (tr - tl)

        def binary_search(i, j):
            # binary_search to find a good starting point
            if i == j:
                return i
            m = int(i + (j - i) / 2)
            interval = self.mapping[col][m]
            if left >= interval.right:
                return binary_search(m, j)
            elif right <= interval.left:
                return binary_search(i, m)
            else:
                return m

        (left, right) = range
        if left is None: left = -np.Inf
        if right is None: right = np.Inf
        query = []
        coverage = []
        min_val = min(list(self.mapping[col].keys()))
        max_val = max(list(self.mapping[col].keys()))
        if left >= self.mapping[col][max_val].right or right <= self.mapping[col][min_val].left:
            print(left, self.mapping[col][max_val].right, right, self.mapping[col][min_val].left)
            return None, None
        start_point = binary_search(min_val, max_val)
        start_point_left = start_point
        start_point_right = start_point + 1
        indicator_left = True
        indicator_right = True
        while (start_point_left >= min_val and start_point_right < max_val
               and (indicator_left or indicator_right)):
            if indicator_left:
                cover = cal_coverage(left, right, self.mapping[col][start_point_left])
                if cover != 0:
                    query.append(start_point_left)
                    coverage.append(cover)
                    start_point_left -= 1
                else:
                    indicator_left = False
            if indicator_right:
                cover = cal_coverage(left, right, self.mapping[col][start_point_right])
                if cover != 0:
                    query.append(start_point_right)
                    coverage.append(cover)
                    start_point_right += 1
                else:
                    indicator_right = False
        return query, np.asarray(coverage)

    def one_iter_of_infer(self, query, n_distinct):
        """Performance a BP in random order.
           This adapts the BP implemented in pgympy package itself.
        """
        copy_query = copy.deepcopy(query)
        sampling_order = copy.deepcopy(self.node_names)
        np.random.shuffle(sampling_order)

        p_estimate = 1
        for attr in sampling_order:
            if attr in query:
                val = copy_query.pop(attr)
                probs = self.infer_machine.query([attr], evidence=copy_query).values
                if any(np.isnan(probs)):
                    p_estimate = 0
                    break
                p = probs[val] / (np.sum(probs)) * n_distinct[attr]
                p_estimate *= p

        return p_estimate

    def query_decoding(self, query, coverage=None, epsilon=0.5):
        """
        Convert the query to the encodings BN recognize
        """
        n_distinct = dict()
        for attr in query:
            if self.attr_type[attr] == 'continuous':
                if coverage is None:
                    n_d_temp = None
                    if type(query[attr]) == tuple:
                        l = max(self.domain[attr][0], query[attr][0])
                        r = min(self.domain[attr][1], query[attr][1])
                    else:
                        l = query[attr]-epsilon
                        r = query[attr]+epsilon
                        if attr in self.n_distinct_mapping:
                            if query[attr] in self.n_distinct_mapping[attr]:
                                n_d_temp = self.n_distinct_mapping[attr][query[attr]]
                    if l > r:
                        return None, None
                    query[attr], n_distinct[attr] = self.continuous_range_map(attr, (l, r))
                    if query[attr] is None:
                        return None, None
                    if n_d_temp is not None:
                        n_distinct[attr] *= n_d_temp
                else:
                    n_distinct[attr] = coverage[attr]
            elif type(query[attr]) == tuple:
                query_list = []
                if self.null_values is None or len(self.null_values) == 0 or attr not in self.null_values:
                    for val in self.encoding[attr]:
                        if query[attr][0] <= val <= query[attr][1]:
                            query_list.append(val)
                else:
                    for val in self.encoding[attr]:
                        if val != self.null_values[attr] and query[attr][0] <= val <= query[attr][1]:
                            query_list.append(val)
                encode_value = self.apply_encoding_to_value(query_list, attr)
                if encode_value is None or (encode_value == []):
                    return None, None
                n_distinct[attr] = self.apply_ndistinct_to_value(encode_value, query_list, attr)
                query[attr], n_distinct[attr] = self.realign(encode_value, n_distinct[attr])
            else:
                encode_value = self.apply_encoding_to_value(query[attr], attr)
                if encode_value is None or (encode_value == []):
                    return None, None
                n_distinct[attr] = self.apply_ndistinct_to_value(encode_value, query[attr], attr)
                query[attr], n_distinct[attr] = self.realign(encode_value, n_distinct[attr])
        return query, n_distinct


    def align_cpds_in_topological(self):
        cpds = self.model.cpds
        sampling_order = []
        while len(sampling_order) < len(self.structure):
            for i, deps in enumerate(self.structure):
                if i in sampling_order:
                    continue  # already ordered
                if all(d in sampling_order for d in deps):
                    sampling_order.append(i)
        topological_order = sampling_order
        topological_order_node = [self.node_names[i] for i in sampling_order]
        new_cpds = []
        for n in topological_order_node:
            for cpd in cpds:
                if cpd.variable == n:
                    new_cpds.append(cpd)
                    break
        assert len(cpds) == len(new_cpds)
        return new_cpds, topological_order, topological_order_node


    def get_condition(self, evidence, cpd, topological_order_node, var_evidence, n_distinct=None, hard_sample=False):
        values = cpd.values
        if evidence[0][0] == -1:
            assert len(values.shape) == 1
            if n_distinct:
                probs = values[var_evidence] * n_distinct
            else:
                probs = values[var_evidence]
            return_prob = np.sum(probs)
            probs = probs / return_prob  # re-normalize
            new_evidence = np.random.choice(var_evidence, p=probs, size=evidence.shape[-1])
        else:
            scope = cpd.variable
            condition = cpd.variables[1:]
            condition_ind = [topological_order_node.index(c) for c in condition]
            condition_evidence = evidence[condition_ind]
            # the following is hardcoded for fast computation
            if len(condition) == 1:
                probs = values[:, condition_evidence[0]]
            elif len(condition) == 2:
                probs = values[:, condition_evidence[0], condition_evidence[1]]
            elif len(condition) == 3:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2]]
            elif len(condition) == 4:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2],
                        condition_evidence[3]]
            elif len(condition) == 5:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2],
                        condition_evidence[3], condition_evidence[4]]
            elif len(condition) == 6:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2],
                        condition_evidence[3], condition_evidence[4], condition_evidence[5]]
            elif len(condition) == 7:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2],
                        condition_evidence[3], condition_evidence[4], condition_evidence[5],
                        condition_evidence[6]]
            else:
                # no more efficient tricks
                probs = np.zeros((values.shape[0], evidence.shape[-1]))
                for j in range(values.shape[0]):
                    probs[j, :] = values[j]
            #print(len(var_evidence))
            #print(probs.shape)
            if n_distinct:
                probs = (probs[var_evidence, :].transpose() * n_distinct).transpose()
            else:
                probs = probs[var_evidence, :]
            #print(probs.shape)
            return_prob = np.sum(probs, axis=0)
            #print(return_prob.shape)
            probs = (probs / return_prob)
            probs[np.isnan(probs)] = 0
            if hard_sample:
                probs += 1e-7
                probs = probs/np.sum(probs, axis=0)
                new_evidence = np.asarray([np.random.choice(var_evidence, p=probs[:,i]) for i in range(evidence.shape[-1])])
                #print(probs.shape)
            else:
                generate_probs = probs.mean(axis=1)
                if np.sum(generate_probs) == 0:
                    return 0, None
                generate_probs = generate_probs / np.sum(generate_probs)
                new_evidence = np.random.choice(var_evidence, p=generate_probs, size=evidence.shape[-1])
        return return_prob, new_evidence


    def query(self, query, num_samples=1, n_distinct=None, coverage=None, return_prob=False, sample_size=1000,
              hard_sample=False):
        """Probability inference using Loopy belief propagation. For example estimate P(X=x, Y=y, Z=z)
           ::Param:: query: dictionary of the form {X:x, Y:y, Z:z}
                     x,y,z can only be a single value
                     num_samples: how many times to run inference, only useful for approaximate algo
                     an approaximation, we might to run it for multiple times and take the average.
                     coverage: the same as ndistinct for continous data
                     return_prob: if true, return P(X=x, Y=y, Z=z)
                                  else return P(X=x, Y=y, Z=z)*nrows
        """
        assert self.infer_algo is not None, "must call .init_inference_method() first"
        if self.infer_algo == "sampling":
            p_estimate = self.query_sampling(query, sample_size)
            if return_prob:
                return (p_estimate, self.nrows)
            return p_estimate * self.nrows
        
        if len(query) == 0:
            if return_prob:
                return 1, self.nrows
            else:
                return self.nrows

        nrows = self.nrows
        if n_distinct is None:
            query, n_distinct = self.query_decoding(query, coverage)
        #print(f"decoded query is {query}")
        if query is None:
            if return_prob:
                return 0, nrows
            else:
                return 0

        if self.infer_algo == "progressive_sampling":
            p_estimate = self.progressive_sampling(query, sample_size, n_distinct, hard_sample=hard_sample)
            if return_prob:
                return (p_estimate, self.nrows)
            return p_estimate * self.nrows

        elif self.infer_algo == "exact-jit" or self.infer_algo == "exact-jit-torch":
            p_estimate = self.infer_machine.query(query, n_distinct)
            if return_prob:
                return (p_estimate, self.nrows)
            return p_estimate * self.nrows

        elif self.infer_algo == "exact" or num_samples == 1:
            sampling_order = list(query.keys())
            p_estimate = 1
            for attr in sampling_order:
                if attr in query:
                    val = query.pop(attr)
                    probs = self.infer_machine.query([attr], evidence=query).values
                    if np.any(np.isnan(probs)):
                        p_estimate = 0
                        break
                    p = np.sum(probs[val] * n_distinct[attr])
                    p_estimate *= p
        else:
            p_estimates = []
            for i in range(num_samples):
                p_estimates.append(self.one_iter_of_infer(query, n_distinct))
            p_estimate = sum(p_estimates) / num_samples

        if return_prob:
            return (p_estimate, nrows)
        return round(p_estimate * nrows)

    def query_id_prob(self, query, id_attrs, n_distinct=None, return_prob=False, reshape=True):
        """
        Calculating the probability distribution of id P(id|Q) * |Q|
        Parameters
        ----------
        Rest parameters: the same as previous function .query().
        """

        if self.infer_algo == "exact-jit" or self.infer_algo == "exact-jit-torch":
            if n_distinct is None:
                query, n_distinct = self.query_decoding(query)
            id_attrs, temp = self.infer_machine.query_id_prob(query, id_attrs, n_distinct)
            if reshape:
                if len(id_attrs) == 1:
                    res = np.zeros(self.bin_size[id_attrs[0]])
                    if self.id_exist_null[id_attrs[0]]:
                        res[self.id_value_position[id_attrs[0]]] = temp[1:]

                    else:
                        res[self.id_value_position[id_attrs[0]]] = temp
                elif len(id_attrs) == 2:
                    res = np.zeros((self.bin_size[id_attrs[0]], self.bin_size[id_attrs[1]]))

                    indx1 = self.id_value_position[id_attrs[0]]
                    indx2 = self.id_value_position[id_attrs[1]]
                    if self.id_exist_null[id_attrs[0]] and self.id_exist_null[id_attrs[1]]:
                        res[np.ix_(indx1, indx2)] = temp[1:, 1:]
                    elif self.id_exist_null[id_attrs[0]]:
                        res[np.ix_(indx1, indx2)] = temp[1:, :]
                    elif not self.id_exist_null[id_attrs[0]] and self.id_exist_null[id_attrs[1]]:
                        res[np.ix_(indx1, indx2)] = temp[:, 1:]
                    else:
                        res[np.ix_(indx1, indx2)] = temp
                else:
                    # TODO: conditional independence exploration and simplification
                    assert False, "for getting the probability of more than three ids, we need to " \
                                  "explore their conditional independence and simplify the structure" \
                                  "to avoid memory explosion. The authors are currently working on it."

            else:
                res = temp
            if return_prob:
                return id_attrs, res
            else:
                return id_attrs, res * self.nrows

    def legitimacy_check(self):
        """
        Checking whether a BN is legitimate
        """
        # Step 1: checking the attrs
        attr_names = list(self.attr_type.keys())
        for col in attr_names:
            if col in self.id_attributes:
                continue
            if self.attr_type[col] == "boolean":
                assert self.mapping[col] is None or len(
                    self.mapping[col]) == 0, f"mapping is for continuous values only"
                assert self.n_in_bin[col] is None or len(
                    self.n_in_bin[col]) == 0, f"n_in_bin is for categorical values only"
            elif self.attr_type[col] == "categorical":
                assert self.mapping[col] is None or len(
                    self.mapping[col]) == 0, f"[{col}] mapping is for continuous values only"
                reverse_encoding = dict()
                for k in self.encoding[col]:
                    enc = self.encoding[col][k]
                    if enc in reverse_encoding:
                        reverse_encoding[enc].append(k)
                    else:
                        reverse_encoding[enc] = [k]
                for enc in self.n_in_bin[col]:
                    assert enc in reverse_encoding, f"{enc} in {col} in n_in_bin is not a valid encoding"
                    n_in_bin_keys = set(list(self.n_in_bin[col][enc].keys()))
                    reverse_keys = set(reverse_encoding[enc])
                    assert n_in_bin_keys == reverse_keys, f"{col} has n_in_bin and encoding mismatch"
            elif self.attr_type[col] == "continuous":
                assert self.encoding[col] is None or len(
                    self.encoding[col]) == 0, f"encoding is for categorical values only"
                assert self.n_in_bin[col] is None or len(
                    self.n_in_bin[col]) == 0, f"n_in_bin is for categorical values only"
                prev = None
                for enc in self.mapping[col]:
                    interval = self.mapping[col][enc]
                    if prev:
                        assert interval.right > prev, f"{col} has unordered intervals for continuous variable"
                    else:
                        prev = interval.right
            else:
                assert False, f"Unknown column type {self.attr_type[col]}"

        # Step 2: checking the CPDs
        for cpd in self.model.cpds:
            for col in cpd.state_names:
                assert col in self.attr_type, f"column {col} not found"
                if self.attr_type[col] == "continuous":
                    mapping = set(list(self.mapping[col].keys()))
                    assert mapping == set(cpd.state_names[col]), f"{col} does not have correct mapping"
                else:
                    if col in self.id_attributes:
                        continue
                    encoding = set(list(self.encoding[col].values()))
                    assert encoding == set(cpd.state_names[col]), f"{col} does not have correct encoding"


