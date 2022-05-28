import sqlalchemy


def create_table(metadata):
    users = sqlalchemy.Table(
        "User_Details",
        metadata,
        sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column("username", sqlalchemy.String),
        sqlalchemy.Column("email", sqlalchemy.String),
        sqlalchemy.Column("designation", sqlalchemy.String)
    )
    return users
