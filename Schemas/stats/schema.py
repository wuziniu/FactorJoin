from Schemas.graph_representation import SchemaGraph, Table

def gen_stats_old_schema(hdf_path):
    """
    Generate the stats schema with a small subset of data.
    """

    schema = SchemaGraph()

    # tables

    # badges
    schema.add_table(Table('badges', attributes=['Id', 'UserId', 'Date'],
                           irrelevant_attributes=['Id'],
                           no_compression=[],
                           csv_file_location=hdf_path.format('badges'),
                           table_size=79851))

    # votes
    schema.add_table(Table('votes', attributes=['Id', 'PostId', 'VoteTypeId', 'CreationDate', 'UserId',
                                                'BountyAmount'],
                           csv_file_location=hdf_path.format('votes'),
                           irrelevant_attributes=['Id', 'BountyAmount'],
                           no_compression=['VoteTypeId'],
                           table_size=328064))

    # postHistory
    schema.add_table(Table('postHistory', attributes=['Id', 'PostHistoryTypeId', 'PostId', 'CreationDate',
                                                      'UserId'],
                           csv_file_location=hdf_path.format('postHistory'),
                           irrelevant_attributes=['Id'],
                           no_compression=['PostHistoryTypeId'],
                           table_size=303187))

    # posts
    schema.add_table(Table('posts', attributes=['Id', 'PostTypeId', 'CreationDate',
                                                'Score', 'ViewCount', 'OwnerUserId',
                                                'AnswerCount', 'CommentCount', 'FavoriteCount',
                                                'LastEditorUserId'],
                           csv_file_location=hdf_path.format('posts'),
                           irrelevant_attributes=['LastEditorUserId'],
                           no_compression=['PostTypeId'],
                           table_size=91976))

    # users
    schema.add_table(Table('users', attributes=['Id', 'Reputation', 'CreationDate', 'Views', 'UpVotes', 'DownVotes'],
                           csv_file_location=hdf_path.format('users'),
                           no_compression=[],
                           table_size=40325))

    # comments
    schema.add_table(Table('comments', attributes=['Id', 'PostId', 'Score', 'CreationDate', 'UserId'],
                           csv_file_location=hdf_path.format('comments'),
                           irrelevant_attributes=['Id'],
                           no_compression=[],
                           table_size=174305))

    # postLinks
    schema.add_table(Table('postLinks', attributes=['Id', 'CreationDate', 'PostId', 'RelatedPostId', 'LinkTypeId'],
                           csv_file_location=hdf_path.format('postLinks'),
                           irrelevant_attributes=['Id'],
                           no_compression=["LinkTypeId"],
                           table_size=11102))

    # tags
    schema.add_table(Table('tags', attributes=['Id', 'Count', 'ExcerptPostId'],
                           csv_file_location=hdf_path.format('tags'),
                           irrelevant_attributes=[],
                           no_compression=["LinkTypeId"],
                           table_size=1032))


    # relationships
    schema.add_relationship('comments', 'PostId', 'posts', 'Id')
    schema.add_relationship('comments', 'UserId', 'users', 'Id')

    schema.add_relationship('badges', 'UserId', 'users', 'Id')

    schema.add_relationship('tags', 'ExcerptPostId', 'posts', 'Id')

    schema.add_relationship('postLinks', 'PostId', 'posts', 'Id')
    schema.add_relationship('postLinks', 'RelatedPostId', 'posts', 'Id')

    schema.add_relationship('postHistory', 'PostId', 'posts', 'Id')
    schema.add_relationship('postHistory', 'UserId', 'users', 'Id')

    schema.add_relationship('votes', 'PostId', 'posts', 'Id')
    schema.add_relationship('votes', 'UserId', 'users', 'Id')

    schema.add_relationship('posts', 'OwnerUserId', 'users', 'Id')
    schema.add_relationship('posts', 'LastEditorUserId', 'users', 'Id')

    return schema


def gen_stats_light_schema(hdf_path):
    """
    Generate the stats schema with a small subset of data.
    """

    schema = SchemaGraph()

    # tables

    # badges
    schema.add_table(Table('badges',
                           primary_key=["Id"],
                           attributes=['Id', 'UserId', 'Date'],
                           irrelevant_attributes=['Id'],
                           no_compression=[],
                           csv_file_location=hdf_path.format('badges'),
                           table_size=79851))

    # votes
    schema.add_table(Table('votes',
                           primary_key=["Id"],
                           attributes=['Id', 'PostId', 'VoteTypeId', 'CreationDate', 'UserId', 'BountyAmount'],
                           csv_file_location=hdf_path.format('votes'),
                           irrelevant_attributes=['Id'],
                           no_compression=['VoteTypeId'],
                           table_size=328064))

    # postHistory
    schema.add_table(Table('postHistory',
                           primary_key=["Id"],
                           attributes=['Id', 'PostHistoryTypeId', 'PostId', 'CreationDate', 'UserId'],
                           csv_file_location=hdf_path.format('postHistory'),
                           irrelevant_attributes=['Id'],
                           no_compression=['PostHistoryTypeId'],
                           table_size=303187))

    # posts
    schema.add_table(Table('posts',
                           primary_key=["Id"],
                           attributes=['Id', 'PostTypeId', 'CreationDate',
                                       'Score', 'ViewCount', 'OwnerUserId',
                                       'AnswerCount', 'CommentCount', 'FavoriteCount', 'LastEditorUserId'],
                           csv_file_location=hdf_path.format('posts'),
                           irrelevant_attributes=['LastEditorUserId'],
                           no_compression=['PostTypeId'],
                           table_size=91976))

    # users
    schema.add_table(Table('users',
                           primary_key=["Id"],
                           attributes=['Id', 'Reputation', 'CreationDate', 'Views', 'UpVotes', 'DownVotes'],
                           csv_file_location=hdf_path.format('users'),
                           no_compression=[],
                           table_size=40325))

    # comments
    schema.add_table(Table('comments',
                           primary_key=["Id"],
                           attributes=['Id', 'PostId', 'Score', 'CreationDate', 'UserId'],
                           csv_file_location=hdf_path.format('comments'),
                           irrelevant_attributes=["Id"],
                           no_compression=[],
                           table_size=174305))

    # postLinks
    schema.add_table(Table('postLinks',
                           primary_key=["Id"],
                           attributes=['Id', 'CreationDate', 'PostId', 'RelatedPostId', 'LinkTypeId'],
                           csv_file_location=hdf_path.format('postLinks'),
                           irrelevant_attributes=["Id"],
                           no_compression=[],
                           table_size=11102))

    # tags
    schema.add_table(Table('tags', attributes=['Id', 'Count', 'ExcerptPostId'],
                           csv_file_location=hdf_path.format('tags'),
                           irrelevant_attributes=["Id"],
                           no_compression=[],
                           table_size=1032))


    # relationships
    schema.add_relationship('comments', 'PostId', 'posts', 'Id')
    schema.add_relationship('comments', 'UserId', 'users', 'Id')

    schema.add_relationship('badges', 'UserId', 'users', 'Id')

    schema.add_relationship('tags', 'ExcerptPostId', 'posts', 'Id')

    schema.add_relationship('postLinks', 'PostId', 'posts', 'Id')
    schema.add_relationship('postLinks', 'RelatedPostId', 'posts', 'Id')

    schema.add_relationship('postHistory', 'PostId', 'posts', 'Id')
    schema.add_relationship('postHistory', 'UserId', 'users', 'Id')
    schema.add_relationship('votes', 'PostId', 'posts', 'Id')
    schema.add_relationship('votes', 'UserId', 'users', 'Id')

    schema.add_relationship('posts', 'OwnerUserId', 'users', 'Id')

    return schema



