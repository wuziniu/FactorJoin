import numpy as np
import pandas as pd
from Schemas.imdb.schema import gen_job_light_imdb_schema

def generate_single_queries(df, attr_type, table_name='title', abb = 't'):
    query = f"SELECT COUNT(*) FROM {table_name} {abb} WHERE "
    column_names = df.columns
    sub_df = None
    n_cols = 0
    for col in column_names:
        a = np.random.choice(3)
        if a == 0:
            if attr_type[col] == "discrete":
                index = np.random.choice(len(df), size=2)
                val = list(set(df[col].iloc[index]))
                if len(val) == 1:
                    sub_query = col + ' == ' + str(val[0])
                else:
                    sub_query = col + ' in ' + str(val)
            elif attr_type[col] == "continuous":
                index = np.random.choice(len(df), size=20)
                val = sorted(list(df[col].iloc[index]))
                choose_idx = np.random.choice(19)
                left_val = val[choose_idx]
                right_val = val[choose_idx+1]
                sub_query = str(left_val) + ' < ' + col + '>' + str(right_val)
            if n_cols == 0:
                sub_df = df.query(sub_query)
                if len(sub_df) == 0:
                    return None
                query += (abb+"."+sub_query)
            else:
                sub_df = sub_df.query(sub_query)
                if len(sub_df) == 0:
                    return None
                query += (' AND '+abb+"."+sub_query)
            n_cols += 1
    if sub_df is None:
        return None
    print(len(sub_df))
    query += ('||'+str(len(sub_df))+'\n')
    return query

def save_ground_true(df, attr_type, save_file = './imdb_single/query.txt'):
    with open(save_file, "a") as text_file:
        for i in range(400):
            q = generate_single_queries(df, attr_type)
            print(i, q)
            if q is not None:
                text_file.write(q)

def run_imdb_single_title():
    file_name = '/home/ziniu.wzn/imdb-benchmark/title.csv'
    df = pd.read_csv(file_name, sep=',', header=0)
    save_ground_true(df)

def get_table(rels, schema, abbs):
    rel_query = ""
    tables = set()
    keys = dict()
    for rel in rels:
        relation = schema.relationship_dictionary[rel]
        tables.add(relation.start)
        keys[relation.start] = relation.start_attr
        tables.add(relation.end)
        keys[relation.end] = relation.end_attr
        query = f"{abbs[relation.start]}.{relation.start_attr}={abbs[relation.end]}.{relation.end_attr}"
        rel_query += (query + " AND ")
    return list(tables), keys, rel_query


def generate_one_table(df, attr_type, table_obj, column_names, table_name='title', abb='t'):
    query = ""
    new_cols = []
    sub_df = None
    n_cols = 0
    for col in column_names:
        col_name = table_name+"__"+col
        if attr_type[col] == "discrete":
            index = np.random.choice(len(df), size=2)
            val = list(set(df[col_name].iloc[index]))
            while any(np.isnan(val)):
                index = np.random.choice(len(df), size=2)
                val = list(set(df[col_name].iloc[index]))
            print(val)
            if len(val) == 1:
                sub_query = col_name + ' == ' + str(val[0])
                act_sub_query = abb+"."+col + '=' + str(val[0]) + ' AND '
            else:
                sub_query = col_name + ' in ' + str(val)
                act_sub_query=""
                for v in val:
                    act_sub_query += (abb+"." + col + '=' + str(v) + ' AND ')
        elif attr_type[col] == "continuous":
            index = np.random.choice(len(df), size=20)
            val = sorted(list(df[col_name].iloc[index]))
            choose_idx = np.random.choice(19)
            left_val = val[choose_idx]
            right_val = val[choose_idx+1]
            sub_query = str(left_val) + ' < ' + col_name + '>' + str(right_val)
            act_sub_query = col + '>' + str(left_val) + ' AND ' + col + '<' + str(right_val)
        if n_cols == 0:
            sub_df = df.query(sub_query)
            if len(sub_df) == 0:
                return None, None, None
            query += act_sub_query
        else:
            sub_df = sub_df.query(sub_query)
            if len(sub_df) == 0:
                return None, None, None
            query += act_sub_query
        n_cols += 1
    if sub_df is None:
        return None, None, None
    print(len(sub_df))
    return sub_df, query, len(sub_df)


def generate_join_query(tables, abbs, schema, attr_types, column_names):
    rels = []
    for relationship_obj in schema.relationships:
        rels.append(relationship_obj.identifier)
    #n_rel = np.random.choice([1, 2], p=[0.8, 0.2])
    selected_rel = list(np.random.choice(rels, size=1, replace=False))
    print(selected_rel)
    selected_table, keys, rel_query = get_table(selected_rel, schema, abbs)
    table_str = ""
    query_str = ""
    sub_df = None
    table_str += f" title {abbs['title']},"
    (sub_df, sub_query, card) = generate_one_table(tables['title'], attr_types['title'],
                               schema.table_dictionary['title'], column_names['title'], 'title', abbs['title'])
    if sub_df is None:
        return None
    else:
        query_str += sub_query

    for table_name in selected_table:
        if table_name != "title":
            table_str += f" {table_name} {abbs[table_name]}"
            sub_df = pd.merge(sub_df, tables[table_name], left_on='title__id',
                              right_on=table_name+'__'+keys[table_name], how='outer')
            (sub_df, sub_query, card) = generate_one_table(sub_df, attr_types[table_name],
                                schema.table_dictionary[table_name], column_names[table_name],
                                                           table_name, abbs[table_name])
            if sub_df is None:
                return None
            else:
                query_str += sub_query

    query = f"SELECT COUNT(*) FROM{table_str} WHERE {rel_query}{query_str[0:-5]}||{card}\n"
    return query

def run_imdb_light():
    schema = gen_job_light_imdb_schema(" ")
    folder = "/home/ziniu.wzn/imdb-benchmark/gen_single_light/"
    all_df = dict()
    abbs = {'title': 't', 'movie_info': 'mi', 'movie_keyword': 'mk', 'movie_info_idx': 'mi_idx',
            'cast_info': 'ci', 'movie_companies': 'mc'}
    attr_types = dict()
    for table in schema:
        df = pd.read_hdf(folder+f"{table}.hdf")
        all_df[table] = df
        table_obj = schema.table_dictionary[table]
        column_names = [col for col in table_obj.attributes if col not in table_obj.irrelevant_attributes]
        attr = dict()
        for col in column_names:
            attr[col] = 'discrete'
        attr_types[table] = attr
    column_names = {
        'title': ['production_year', 'kind_id'],
        'movie_info_idx': ['info_type_id'],
        'movie_info': ['info_type_id'],
        'cast_info': ['role_id'],
        'movie_companies': ['company_type_id'],
        'movie_keyword': []
    }
    save_file = './benchmark/job-light/join_query.txt'
    with open(save_file, "a") as text_file:
        for i in range(400):
            q = generate_join_query(all_df, abbs, schema, attr_types, column_names)
            print(i, q)
            if q is not None:
                text_file.write(q)

