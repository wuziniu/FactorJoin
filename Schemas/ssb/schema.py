from Schemas.graph_representation import SchemaGraph, Table


def gen_1gb_ssb_schema(csv_path):
    """
    SSB schema for SF=1.
    """

    schema = SchemaGraph()

    # tables
    # lineorder
    schema.add_table(Table('lineorder',
                           attributes=['lo_orderkey', 'lo_linenumber', 'lo_custkey', 'lo_partkey', 'lo_suppkey',
                                       'lo_orderdate', 'lo_orderpriority', 'lo_shippriority', 'lo_quantity',
                                       'lo_extendedprice', 'lo_ordertotalprice', 'lo_discount', 'lo_revenue',
                                       'lo_supplycost', 'lo_tax', 'lo_commitdate', 'lo_shipmode'],
                           irrelevant_attributes=['lo_commitdate'],
                           csv_file_location=csv_path.format('lineorder_sampled'),
                           table_size=6001171,
                           primary_key=[]
                           ))

    # dwdate
    # dwdate.d_dayofweek -> dwdate.d_daynuminweek
    # dwdate.d_dayofweek -> dwdate.d_lastdayinweekfl
    # dwdate.d_month -> dwdate.d_monthnuminyear
    # dwdate.d_monthnuminyear -> dwdate.d_sellingseason
    # dwdate.d_daynuminyear -> dwdate.d_weeknuminyear
    schema.add_table(
        Table('date',
              attributes=['d_datekey', 'd_date', 'd_dayofweek', 'd_month', 'd_year', 'd_yearmonthnum', 'd_yearmonth',
                          'd_daynuminweek', 'd_daynuminmonth', 'd_daynuminyear', 'd_monthnuminyear', 'd_weeknuminyear',
                          'd_sellingseason', 'd_lastdayinweekfl', 'd_lastdayinmonthfl', 'd_holidayfl', 'd_weekdayfl'],
              irrelevant_attributes=['d_date'],
              csv_file_location=csv_path.format('date'),
              table_size=2556,
              primary_key=["d_datekey"]
              ))

    # customer
    # customer.c_city -> customer.c_nation
    # customer.c_nation -> customer.c_region
    schema.add_table(
        Table('customer',
              attributes=['c_custkey', 'c_name', 'c_address', 'c_city', 'c_nation', 'c_region', 'c_phone',
                          'c_mktsegment'],
              irrelevant_attributes=['c_name', 'c_address', 'c_phone'],
              csv_file_location=csv_path.format('customer'),
              table_size=30000,
              primary_key=["c_custkey"]
              ))

    # part
    # part.p_brand1 -> part.p_category
    # part.p_category -> part.p_mfgr
    schema.add_table(
        Table('part',
              attributes=['p_partkey', 'p_name', 'p_mfgr', 'p_category', 'p_brand1', 'p_color', 'p_type', 'p_size',
                          'p_container'],
              irrelevant_attributes=['p_name'],
              csv_file_location=csv_path.format('part'),
              table_size=200000,
              primary_key=["p_partkey"]
              ))

    # supplier
    # supplier.s_city -> supplier.s_nation
    # supplier.s_nation -> supplier.s_region
    schema.add_table(
        Table('supplier', attributes=['s_suppkey', 's_name', 's_address', 's_city', 's_nation', 's_region', 's_phone'],
              irrelevant_attributes=['s_name', 's_address', 's_phone'],
              csv_file_location=csv_path.format('supplier'),
              table_size=2000,
              primary_key=["s_suppkey"]))

    # relationships
    schema.add_relationship('lineorder', 'lo_custkey', 'customer', 'c_custkey')
    schema.add_relationship('lineorder', 'lo_partkey', 'part', 'p_partkey')
    schema.add_relationship('lineorder', 'lo_suppkey', 'supplier', 's_suppkey')
    schema.add_relationship('lineorder', 'lo_orderdate', 'date', 'd_datekey')
    return schema
