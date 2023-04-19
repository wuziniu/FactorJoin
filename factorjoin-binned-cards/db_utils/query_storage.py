import klepto
from .utils import *
#from .query_generator import *
#from .query_generator2 import *
#from cardinality_estimation.query import *
from utils.utils import *
import toml
import multiprocessing
from multiprocessing import Pool
# from cardinality_estimation.db import DB
from networkx.readwrite import json_graph

from sql_rep.utils import execute_query
import copy

def get_all_cardinalities(samples, ckey):
    cards = []
    for qrep in samples:
        for node, info in qrep["subset_graph"].nodes().items():
            if node == SOURCE_NODE:
                continue
            cards.append(info[ckey]["actual"])
            if cards[-1] == 0:
                # print(qrep["sql"])
                # print(node)
                # print(qrep["template_name"])
                # print(info["cardinality"])
                assert False
    return cards

def get_all_totals(qrep):
    '''
    @ret: dict, nodes : total
    '''
    user, pwd, db, db_host, port = get_default_con_creds()
    ret = {}
    par_args = []
    par_subsets = []
    for subset, info in qrep["subset_graph"].nodes().items():
        cards = info["cardinality"]
        sg = qrep["join_graph"].subgraph(subset)
        subsql = nx_graph_to_query(sg)
        tsql = get_total_count_query(subsql)
        par_args.append((tsql, user, db_host, port, pwd, db, []))
        par_subsets.append(subset)

    num_processes = multiprocessing.cpu_count()
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(execute_query, par_args)

    for i, res in enumerate(results):
        assert res is not None
        total_count = res[0][0]
        ret[par_subsets[i]] = total_count

    print("executed successfully!")
    return ret

def update_qrep(qrep, total_sample=None):
    '''
    @qrep: sql_query_rep format
    @total_sample: from same template class, so subquery totals should be the
    same for this one.

    Will update qrep in place with "total" values for subset_graph nodes, and
    pred_cols etc. for join_graph nodes.
    '''
    if total_sample is not None:
        # if "total" not in qrep["subset_graph"].nodes()[tuple("t")]["cardinality"]:
        for node in qrep["subset_graph"].nodes():
            total = total_sample["subset_graph"].nodes()[node]["cardinality"]["total"]
            qrep["subset_graph"].nodes()[node]["cardinality"]["total"] = total

    for node in qrep["join_graph"].nodes():
        if "predicates" not in qrep["join_graph"].nodes()[node]:
            join_graph.nodes[node]["pred_cols"] = []
            join_graph.nodes[node]["pred_types"] = []
            join_graph.nodes[node]["pred_vals"] = []
            continue
        subg = qrep["join_graph"].subgraph(node)
        node_sql = nx_graph_to_query(subg)
        pred_cols, pred_types, pred_vals = extract_predicates(node_sql)
        qrep["join_graph"].nodes[node]["pred_cols"] = pred_cols
        qrep["join_graph"].nodes[node]["pred_types"] = pred_types
        qrep["join_graph"].nodes[node]["pred_vals"] = pred_vals

def gen_queries(query_template, num_samples, args):
    '''
    @query_template: dict, or str, as used by QueryGenerator2 or
    QueryGenerator.
    '''
    if isinstance(query_template, dict):
        qg = QueryGenerator2(query_template, args.user, args.db_host, args.port,
                args.pwd, args.db_name)
    elif isinstance(query_template, str):
        qg = QueryGenerator(query_template, args.user, args.db_host, args.port,
                args.pwd, args.db_name)

    gen_sqls = qg.gen_queries(num_samples)
    gen_sqls = remove_doubles(gen_sqls)
    # TODO: remove queries that evaluate to zero
    return gen_sqls

def load_sql_rep(fn, dummy=None):
    assert ".pkl" in fn
    try:
        with open(fn, "rb") as f:
            query = pickle.load(f)
    except:
        print(fn + " failed to load...")
        exit(-1)

    query["subset_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])
    if "subset_graph_paths" in query:
        query["subset_graph_paths"] = \
                nx.OrderedDiGraph(json_graph.adjacency_graph(query["subset_graph_paths"]))

    return query

def save_sql_rep(fn, cur_qrep):
    assert ".pkl" in fn
    # qrep = nx.DiGraph(cur_qrep)
    qrep = copy.deepcopy(cur_qrep)
    qrep["join_graph"] = nx.adjacency_data(qrep["join_graph"])
    qrep["subset_graph"] = nx.adjacency_data(qrep["subset_graph"])
    # if "subset_graph_paths" in qrep:
        # qrep["subset_graph_paths"] = nx.adjacency_data(qrep["subset_graph_paths"])

    with open(fn, "wb") as f:
        pickle.dump(qrep, f)

def nx_graph_to_query_rep(G, true_count, total_count, pg_count):
    '''
    Extracts all the relevant information from the sql_rep format and puts it
    in the cardinality_estimation/Query format, without re-parsing any part of
    the sql.

    FIXME: each table must have aliases
    '''
    froms = []
    conds = []
    pred_cols = []
    pred_vals = []
    pred_types = []
    table_names = []
    aliases = []

    for nd in G.nodes(data=True):
        node = nd[0]
        data = nd[1]
        assert "real_name" in data
        froms.append(ALIAS_FORMAT.format(TABLE=data["real_name"],
                                         ALIAS=node))
        table_names.append(data["real_name"])
        aliases.append(node)

        for pred in data["predicates"]:
            if pred not in conds:
                conds.append(pred)

        pred_cols += data["pred_cols"]
        pred_vals += data["pred_vals"]
        pred_types += data["pred_types"]

    for edge in G.edges(data=True):
        conds.append(edge[2]['join_condition'])

    # preserve order for caching
    froms.sort()
    conds.sort()
    from_clause = " , ".join(froms)
    if len(conds) > 0:
        wheres = ' AND '.join(conds)
        from_clause += " WHERE " + wheres
    count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause)

    ret_query = Query(count_query, pred_cols,
            pred_vals, pred_types, true_count, total_count,
            pg_count, None, None)
    # TODO: add joins too
    ret_query.table_names = table_names
    ret_query.aliases = aliases

    return ret_query

def convert_sql_rep_to_query_rep(qrep):
    jg = qrep["join_graph"]
    query = None
    subqueries = []

    for nodes in qrep["subset_graph"]:
        card = qrep["subset_graph"].nodes()[nodes]["cardinality"]
        sg = jg.subgraph(nodes)
        sq = nx_graph_to_query_rep(sg, card["actual"], card["total"],
            card["expected"])
        subqueries.append(sq)
        if len(sg.nodes()) == len(jg.nodes()):
            query = sq
    assert query is not None
    query.subqueries = subqueries
    return query

def remove_doubles(query_strs):
    doubles = 0
    newq = []
    seen_samples = set()
    for q in query_strs:
        if q in seen_samples:
            doubles += 1
            # print("seen double!")
            # print(q)
            # pdb.set_trace()
            continue
        seen_samples.add(q)
        newq.append(q)

    if doubles > 0:
        print("removed {} doubles".format(doubles))
    return newq

def gen_query_strs(args, query_template, num_samples,
        sql_str_cache, save_cur_cache_dir=None):
    '''
    @query_template: str OR dict.

    @ret: [Query, Query, ...]
    '''
    query_strs = []

    # TODO: change key to be based on file name?
    if isinstance(query_template, str):
        hashed_tmp = deterministic_hash(query_template)
    elif isinstance(query_template, dict):
        hashed_tmp = deterministic_hash(query_template["base_sql"]["sql"])
    else:
        assert False

    if hashed_tmp in sql_str_cache.archive:
        query_strs = sql_str_cache.archive[hashed_tmp]
        print("loaded {} query strings".format(len(query_strs)))

    if num_samples == -1:
        # select whatever we loaded
        query_strs = query_strs
    elif len(query_strs) > num_samples:
        query_strs = query_strs[0:num_samples]
    elif len(query_strs) < num_samples:
        # need to generate additional queries
        req_samples = num_samples - len(query_strs)
        num_processes = multiprocessing.cpu_count()
        num_processes = min(num_processes, args.num_samples_per_template)
        num_per_p = int(args.num_samples_per_template / num_processes)

        with Pool(processes=num_processes) as pool:
            par_args = [(query_template, num_per_p, args)
                    for _ in range(num_processes)]
            comb_query_strs = pool.starmap(gen_queries, par_args)
        # need to flatten_the list
        for cqueries in comb_query_strs:
            query_strs += cqueries
        print("generated {} query sqls".format(len(query_strs)))

        # if isinstance(query_template, dict):
            # qg = QueryGenerator2(query_template, args.user, args.db_host, args.port,
                    # args.pwd, args.db_name)
        # elif isinstance(query_template, str):
            # qg = QueryGenerator(query_template, args.user, args.db_host, args.port,
                    # args.pwd, args.db_name)

        # gen_sqls = qg.gen_queries(req_samples)
        # query_strs += gen_sqls
        # save on the disk
        sql_str_cache.archive[hashed_tmp] = query_strs

    return query_strs

def gen_query_objs(args, query_strs, query_obj_cache):
    '''
    Note: this must return query objects in the same order as query_strs.
    '''
    ret_queries = []
    unknown_query_strs = []
    idx_map = {}

    # everything below this part is for query objects exclusively
    for i, sql in enumerate(query_strs):
        assert i == len(ret_queries)
        hsql = deterministic_hash(sql)
        if hsql in query_obj_cache:
            curq = query_obj_cache[hsql]
            # if not hasattr(curq, "froms"):
                # print("NEED TO UPDATE QUERY STRUCT")
                # update_query_structure(curq)
                # query_obj_cache.archive[hsql] = curq
            # assert hasattr(curq, "froms")
            # update the query structure as well if needed
            ret_queries.append(curq)
        elif hsql in query_obj_cache.archive:
            try:
                curq = query_obj_cache.archive[hsql]
                # if not hasattr(curq, "froms"):
                    # print("NEED TO UPDATE QUERY STRUCT")
                    # update_query_structure(curq)
                    # query_obj_cache.archive[hsql] = curq
                # assert hasattr(curq, "froms")
                # update the query structure as well if needed
                ret_queries.append(curq)
            except:
                print("klepto query corruption, regenerating...")
                idx_map[len(unknown_query_strs)] = i
                ret_queries.append(None)
                unknown_query_strs.append(sql)
        else:
            idx_map[len(unknown_query_strs)] = i
            ret_queries.append(None)
            unknown_query_strs.append(sql)
            # store the appropriate index

    # print("loaded {} query objects".format(len(ret_queries)))
    if len(unknown_query_strs) == 0:
        return ret_queries
    else:
        print("need to generate {} query objects".\
                format(len(unknown_query_strs)))

    # sql_result_cache = args.cache_dir + "/sql_result"
    sql_result_cache = None
    all_query_objs = []
    start = time.time()
    # num_processes = int(min(len(unknown_query_strs),
        # multiprocessing.cpu_count() / 2))
    num_processes = int(min(len(unknown_query_strs),
        multiprocessing.cpu_count()))
    num_processes = max(num_processes, 1)
    with Pool(processes=num_processes) as pool:
        args = [(cur_query, args.user, args.db_host, args.port,
            args.pwd, args.db_name, None,
            args.execution_cache_threshold, sql_result_cache, int(1800000/2), i) for
            i, cur_query in enumerate(unknown_query_strs)]
        all_query_objs = pool.starmap(sql_to_query_object, args)

    for i, q in enumerate(all_query_objs):
        ret_queries[idx_map[i]] = q
        hsql = deterministic_hash(unknown_query_strs[i])
        # save in memory, so potential repeat queries can be found in the
        # memory cache
        query_obj_cache[hsql] = q
        # save at the disk backend as well, without needing to dump all of
        # the cache
        query_obj_cache.archive[hsql] = q

    print("generated {} samples in {} secs".format(len(unknown_query_strs),
        time.time()-start))

    assert len(ret_queries) == len(query_strs)

    # sanity check: commented out so we don't spend time here
    # for i, query in enumerate(ret_queries):
        # assert query.query == query_strs[i]

    # why were we doing this anyway?
    # for i, query in enumerate(ret_queries):
        # ret_queries[i] = Query(query.query, query.pred_column_names,
                # query.vals, query.cmp_ops, query.true_count, query.total_count,
                # query.pg_count, query.pg_marginal_sels, query.marginal_sels)

    return ret_queries

def get_template_samples(fn):
    # number of samples to use from this template (fn)
    if "2.toml" in fn:
        num = 500
    elif "2b1.toml" in fn:
        num = 500
    elif "2b2.toml" in fn:
        num = 500
    elif "2b3.toml" in fn:
        num = 500
    elif "2b4.toml" in fn:
        num = 500
    elif "2d2.toml" in fn:
        num = 730
    elif "2d.toml" in fn:
        num = 900
    elif "2dtitle.toml" in fn:
        num = 298
    elif "2U2.toml" in fn:
        num = 1159
    elif "2U3.toml" in fn:
        num = 665
    elif "4.toml" in fn:
        num = 1383
    elif "3.toml" in fn:
        num = 100
    elif "7.toml" in fn:
        num = 170
    elif "7b.toml" in fn:
        num = 560
    elif "8.toml" in fn:
        num = 548
    elif "7c.toml" in fn:
        num = 20
    elif "5.toml" in fn:
        num = 516
    elif "6.toml" in fn:
        num = 1017
    elif "9.toml" in fn:
        num = 180
    else:
        assert False

    return num

def _load_subqueries(args, queries, sql_str_cache, subq_cache,
        gen_subqueries):
    '''
    @ret:
    '''
    start = time.time()
    all_sql_subqueries = []
    new_queries = []

    for i, q in enumerate(queries):
        hashed_key = deterministic_hash(q.query)
        if hashed_key in sql_str_cache:
            assert False
            sql_subqueries = sql_str_cache[hashed_key]
        elif hashed_key in sql_str_cache.archive:
            sql_subqueries = sql_str_cache.archive[hashed_key]
            if not gen_subqueries:
                all_subq_present = True
                for subq_sql in sql_subqueries:
                    hsql = deterministic_hash(subq_sql)
                    if not hsql in subq_cache.archive:
                        all_subq_present = False
                        break
                if not all_subq_present:
                    print("skipping query {} {}".format(q.template_name, i))
                    continue
        else:
            if not gen_subqueries:
                # print("gen_queries is false, so skipping subquery gen")
                continue
            else:
                s1 = time.time()
                print("going to generate subqueries for query num ", i)
                sql_subqueries = gen_all_subqueries(q.query)
                # save it for the future!
                sql_str_cache.archive[hashed_key] = sql_subqueries
                print("generating + saving subqueries: ", time.time() - s1)

        all_sql_subqueries += sql_subqueries
        new_queries.append(q)

    all_subqueries = gen_query_objs(args, all_sql_subqueries, subq_cache)
    print("loop done")
    assert len(all_subqueries) == len(all_sql_subqueries)
    return new_queries, all_sql_subqueries, all_subqueries

def _load_query_strs(args, cache_dir, template, template_fn):
    sql_str_cache = klepto.archives.dir_archive(cache_dir + "/sql_str",
            cached=True, serialized=True)
    # find all the query strs associated with this template
    if args.num_samples_per_template == -1:
        num_samples = get_template_samples(template_fn)
    else:
        num_samples = args.num_samples_per_template
    query_strs = gen_query_strs(args, template,
            num_samples, sql_str_cache)
    return query_strs

def _remove_zero_samples(samples):
    nonzero_samples = []
    for i, s in enumerate(samples):
        if s.true_sel != 0.00:
            nonzero_samples.append(s)
        else:
            pass

    print("len nonzero samples: ", len(nonzero_samples))
    return nonzero_samples

def _save_subq_sqls(queries, subq_sqls, cache_dir):
    sql_cache = klepto.archives.dir_archive(cache_dir + "/subq_sql_str",
            cached=True, serialized=True)
    assert len(queries) == len(subq_sqls)
    for i, q in enumerate(queries):
        sql = q.query
        hkey = deterministic_hash(sql)
        sql_cache.archive[hkey] = subq_sqls[i]

def _save_sqls(template, sqls, cache_dir):
    sql_cache = klepto.archives.dir_archive(cache_dir + "/sql_str",
            cached=True, serialized=True)
    hashed_tmp = deterministic_hash(template["base_sql"]["sql"])
    sql_cache.archive[hashed_tmp] = sqls

def _save_subquery_objs(subqs, cache_dir):
    query_obj_cache = klepto.archives.dir_archive(cache_dir + "/subq_query_obj",
            cached=True, serialized=True)
    for query in subqs:
        hsql = deterministic_hash(query.query)
        query_obj_cache.archive[hsql] = query

def _save_query_objs(queries, cache_dir):
    query_obj_cache = klepto.archives.dir_archive(cache_dir + "/query_obj",
            cached=True, serialized=True)
    for query in queries:
        hsql = deterministic_hash(query.query)
        query_obj_cache.archive[hsql] = query

def _load_query_objs(args, cache_dir, query_strs, template_name=None,
        gen_subqueries=True):
    query_obj_cache = klepto.archives.dir_archive(cache_dir + "/query_obj",
            cached=True, serialized=True)
    samples = gen_query_objs(args, query_strs, query_obj_cache)

    if args.only_nonzero_samples:
        samples = _remove_zero_samples(samples)

    for i, s in enumerate(samples):
        s.template_name = template_name

    return samples

def load_all_queries(args, fn, subqueries=True):
    all_queries = []
    all_subqueries = []

    subq_query_obj_cache = klepto.archives.dir_archive(args.cache_dir +
            "/subq_query_obj", cached=True, serialized=True, memsize=16000)
    subq_sql_str_cache = klepto.archives.dir_archive(args.cache_dir + "/subq_sql_str",
            cached=True, serialized=True)

    assert ".toml" in fn
    template = toml.load(fn)
    query_strs = _load_query_strs(args, args.cache_dir, template, fn)
    # deduplicate
    query_strs = remove_doubles(query_strs)

    template_name = os.path.basename(fn)
    queries = _load_query_objs(args, args.cache_dir, query_strs,
            template_name)

    queries, subq_strs, subqueries = _load_subqueries(args, queries,
            subq_sql_str_cache, subq_query_obj_cache, args.gen_queries)

    print(len(subq_strs), len(queries))
    assert len(subq_strs) % len(queries) == 0
    num_subq_per_query = int(len(subq_strs) / len(queries))
    print("{}: queries: {}, subqueries: {}".format(template_name,
        len(queries), num_subq_per_query))

    start_idx = 0
    for i in range(len(queries)):
        end_idx = start_idx + num_subq_per_query
        all_subqueries.append(subqueries[start_idx:end_idx])
        all_queries.append(queries[i])
        if len(all_subqueries[-1]) == 0:
            print(i)
            print("found no subqueries")
            pdb.set_trace()
        start_idx += num_subq_per_query

    return all_queries, all_subqueries

def update_subq_cards(all_subqueries, cache_dir):
    print("starting update subq cards, this may take a while...")
    for subqueries in all_subqueries:
        wrong_count = 0
        for subq in subqueries:
            if subq.true_count > subq.total_count:
                subq.total_count = subq.true_count
                wrong_count += 1

            if subq.true_sel > 1:
                subq.true_sel = 1.00
                wrong_count += 1

        if wrong_count > 0:
            print("wrong counts: ", wrong_count)
            _save_subquery_objs(subqueries, cache_dir)

def update_subq_preds(all_queries, all_subqueries, cache_dir):
    '''
    @all_queries: list of query objects.
    @all_subqueries: for ith query in all_queries, its list of subquery
    objects.

    Utilizes the fact that parsing the predicate values from queries should
    give us enough information about all the predicate values in subqueries.
    '''
    print("starting update_subq_preds, this may take a while...")
    assert len(all_queries) == len(all_subqueries)
    for i, query in enumerate(all_queries):
        if i % 100 == 0:
            print("updating subqueries predicates for query {}".format(i))
        subqueries = all_subqueries[i]
        if query.pred_column_names is None:
            pred_columns, cmp_ops, pred_vals = extract_predicates(query.query)
            query.pred_column_names = pred_columns
            query.cmp_ops = cmp_ops
            query.vals = pred_vals

        wrong_count = 0

        for subq in subqueries:
            if subq.true_count > subq.total_count:
                subq.total_count = subq.true_count
                wrong_count += 1
            if not hasattr(subq, "aliases"):
                subq.froms, subq.aliases, subq.table_names = extract_from_clause(subq.query)

            if subq.pred_column_names is not None:
                continue
            subq.pred_column_names = []
            subq.cmp_ops = []
            subq.vals = []

            # loop over all the query predicates, and if we find same alias in
            # subq, then add them

            for pi, pred_col in enumerate(query.pred_column_names):
                # is pred_col one of the subq aliases
                pred_table = pred_col[0:pred_col.find(".")]
                if pred_table in subq.aliases:
                    val = query.vals[pi]
                    cmp_op = query.cmp_ops[pi]
                    subq.pred_column_names.append(pred_col)
                    subq.vals.append(val)
                    subq.cmp_ops.append(cmp_op)

            # assert len(subq.pred_column_names) >= 1

        _save_subquery_objs(subqueries, cache_dir)
    _save_query_objs(all_queries, cache_dir)

