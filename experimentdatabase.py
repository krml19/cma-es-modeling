import math
import re
import sqlite3
import sys
import unicodedata

from collections import Iterable
from requests.structures import CaseInsensitiveDict
from sortedcontainers import SortedSet

__author__ = 'Tomasz Pawlak'
__version__ = '1.0.20180320'

# This script requires sqlite3 module shipped with Python 3.6+, as the previous versions automatically commit a transaction before executing a DDL statement
# For details see https://docs.python.org/3.6/library/sqlite3.html#controlling-transactions
# As a version number of sqlite3 module does not change between Pythons 3.5 and 3.6, we have to check both:
assert sys.version_info >= (3, 6)
assert sqlite3.version_info >= (2, 6)


class DataSet(CaseInsensitiveDict):

    @property
    def id(self) -> int:
        self.__check_closed()
        return self.__id

    @property
    def table_name(self) -> str:
        self.__check_closed()
        return self.__table_name

    @property
    def parent(self):
        self.__check_closed()
        return self.__parent

    def __init__(self, name: str, parent, database):
        super().__init__()
        assert name is not None
        assert database is not None

        self.__id = -1
        self.__database = None
        self.__closed = False
        self.__table_name = Helpers.fix_data_object_name(name)
        self.__parent = parent
        self.__database = database
        self.__children = []

    def new_child_data_set(self, table_name: str):
        self.__check_closed()

        child = DataSet(table_name, self, self.__database)
        self.__children.append(child)

        return child

    def __remove_child(self, child):
        self.__children.remove(child)

    def save(self, with_children=True):
        self.__check_closed()

        try:
            self.__database.engine.begin_transaction()

            # determine parent id
            if self.__parent is not None:
                parent_id = self.__parent.id
                if parent_id == -1:
                    self.__parent.save(False)  # gets id from database
                    parent_id = self.__parent.id
                assert parent_id >= 0
                self["parent"] = parent_id

            # add my id to parameters
            if self.__id >= 0:
                self["id"] = self.__id

            # save me
            last_insert_id = self.__database.write_data(self)

            # update my id
            if last_insert_id >= 0:
                assert self.__id == -1 or self.__id == last_insert_id
                self.__id = last_insert_id

            # save my children
            if with_children:
                for child in self.__children:
                    child.save(True)

            self.__database.engine.commit()
        except Exception as ex:
            try:
                self.__database.engine.rollback()
            except:
                pass
            raise DatabaseException("Cannot save DataSet") from ex

    def __check_closed(self):
        if self.__closed:
            raise DatabaseException("The object is already closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.__closed:
            self.save(False)

            while len(self.__children) > 0:
                self.__children[0].__exit__(exc_type, exc_val, exc_tb)  # calls remove_child on self

            if self.__parent is not None:
                assert self in self.__parent.__children
                self.__parent.__remove_child(self)
                assert self not in self.__parent.__children

            self.__closed = True


class Experiment(DataSet):
    def __init__(self, database):
        super().__init__("experiments", None, database)


class Database:
    __CREATE_EXPERIMENTS_TABLE = "CREATE TABLE IF NOT EXISTS experiments(id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT)"

    def __init__(self, database_filename: str):
        try:
            self.__closed = False
            self.engine = DatabaseEngine(database_filename)
            try:
                self.engine.begin_transaction()
                self.__prepare_schema()
                self._columns = self.__get_table_info()
                self.engine.commit()
            except Exception as ex:
                self.engine.rollback()
                raise ex
        except Exception as ex:
            raise DatabaseException("Cannot initialize database.") from ex

    def new_experiment(self) -> Experiment:
        self.__check_closed()
        return Experiment(self)

    def write_data(self, set: DataSet):
        self.__check_closed()
        try:
            self.engine.begin_transaction()
            self.__ensure_columns(set)
            insert = self.__build_insert_statement(set)
            parameters = {col: None if value is float and math.isnan(value) else value for (col, value) in set.items()}
            self.engine.execute(insert, parameters)
            last_insert_id = self.engine.last_insert_id
            self.engine.commit()
            return last_insert_id
        except Exception as ex:
            self.engine.rollback()
            raise ex

    def __prepare_schema(self):
        self.engine.execute(self.__CREATE_EXPERIMENTS_TABLE)

    def __get_table_info(self) -> dict:
        tables = {}
        for row in self.engine.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            name = row["name"]
            tables[name] = self.__get_column_info(name)
        return tables

    def __get_column_info(self, table: str) -> SortedSet:
        columns = SortedSet()
        for row in self.engine.execute("PRAGMA table_info(%s)" % table):
            columns.add(row["name"])
        return columns

    def __check_closed(self):
        if self.__closed:
            raise DatabaseException("Database is already closed")

    def __create_table(self, set: DataSet):
        sql = "CREATE TABLE IF NOT EXISTS `%s`(id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT" % set.table_name
        for col in set:
            if Helpers.equals_ignore_case("id", col):
                continue

            sql += ",`%s`" % col

            if Helpers.equals_ignore_case("parent", col):
                sql += " INTEGER NOT NULL REFERENCES `%s`(id) ON DELETE CASCADE ON UPDATE CASCADE DEFERRABLE INITIALLY DEFERRED" % set.parent.table_name
            else:
                sql += " NUMERIC NULL"
        sql += ")"
        self.engine.execute(sql)
        current_columns = self.__get_column_info(set.table_name)
        self._columns[set.table_name] = current_columns

        return current_columns

    def __ensure_columns(self, set: DataSet):
        """Makes sure that a tables contains columns from a given set. If not, then the columns are created."""

        # This is only a view of database, actually the database table can contain such a column even if the collection not.
        # However if the view contains a column, it is guaranteed that database table has such a column.
        if set.table_name in self._columns:
            current_columns = self._columns[set.table_name]
        else:
            current_columns = self.__create_table(set)
            # Even if table does not exist in database, we cannot assume that creation of a table with all columns
            # specified by the argument will fulfill our requirements. This is because, to suppress errors, we added
            # IF NOT EXISTS clause to the CREATE statement. In case that the table actually exists, it may not
            # have all required columns. So, continue...

        for col in set:
            if col not in current_columns:
                try:
                    # create column in database
                    # Note that the NUMERIC type is only a hint to database to try to convert the given value to
                    # INTEGER or REAL if possible, otherwise the value is stored as supplied (e.g. as TEXT).
                    # INTEGER and REAL values are supposed to consume less storage space than the corresponding
                    # TEXT values. What's important the data type conversions are made mostly implicitly by
                    # database engine executing query. See http://www.sqlite.org/datatype3.html for more details.
                    self.engine.execute("ALTER TABLE `%s` ADD COLUMN `%s` NUMERIC NULL" % (set.table_name, col))
                except Exception:
                    # just ignore it
                    pass
            current_columns.add(col)
            # now the column exists both in our view and in database

    def __build_insert_statement(self, set: DataSet) -> str:
        sql = "INSERT OR REPLACE INTO `%s`" % set.table_name
        if len(set) > 0:
            sql += "("
            for col in set:
                sql += "`%s`," % col

            sql = sql[:-1] + ") VALUES ("
            for col in set:
                sql += ":%s," % col

            sql = sql[:-1] + ")"
        else:
            sql += " DEFAULT VALUES"

        return sql

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__check_closed()
        if not self.__closed:
            # engine may be null if it crashed during this object initialization
            if self.engine is not None:
                self.engine.__exit__(exc_type, exc_val, exc_tb)
                self.__closed = True


class DatabaseEngine:
    __extensions = []

    @property
    def last_insert_id(self) -> int:
        self.__cursor.execute("SELECT last_insert_rowid()")
        return int(self.__cursor.fetchone()[0])

    def __init__(self, database_file):
        self.__closed = False
        self.__current_transaction_level = 0
        # concurrent DEFERRED transactions have problems with locks on Windows (This is strange, as the problem does not happen in Java nor .NET)
        self.__connection = sqlite3.connect(database_file, 1 << 30, isolation_level="IMMEDIATE")
        self.__connection.row_factory = sqlite3.Row
        self.__cursor = self.__connection.cursor()
        self.__cursor.execute("PRAGMA busy_timeout = %d" % (1 << 30))
        assert self.__cursor.execute("PRAGMA busy_timeout").fetchall()[0][0] == 1 << 30, self.__cursor.execute("PRAGMA busy_timeout").fetchall()[0][0]
        # disable foreign key support when filling database due to:
        # 1) performance,
        # 2) ON DELETE clause in foreign key definition, which causes deletions in iterations table when user requires replacement of a tuple in experiment table
        self.__cursor.execute("PRAGMA foreign_keys = 0")
        self.__cursor.execute("PRAGMA synchronous = 0")
        self.__cursor.execute("PRAGMA journal_mode = TRUNCATE")
        self.__cursor.execute("PRAGMA page_size = %d" % (1 << 14))
        self.__cursor.execute("PRAGMA temp_store = MEMORY")

        for ext in self.__extensions:
            raise NotImplementedError("Support for extensions is not implemented yet")

    def begin_transaction(self):
        """Begins database transaction. The subsequent calls to this method increase the level of transactions stack, i.e.
        nested transactions are supported.However the data is guaranteed to be committed to disk, only if the outer most
        transaction is committed.The roll backs of outer transactions also roll back the nested transactions."""
        self.__check_closed()
        if self.__current_transaction_level == 0:
            # concurrent DEFERRED transactions have problems with locks on Windows (This is strange, as the problem does not happen in Java nor .NET)
            self.__cursor.execute("BEGIN IMMEDIATE TRANSACTION")
        else:
            self.__cursor.execute("SAVEPOINT sp_%d" % self.__current_transaction_level)
        self.__current_transaction_level += 1  # in case SAVEPOINT failed

    def commit(self):
        """Commits database transaction. If transactions stack contains more than one transaction, then only the top-level
        transaction is committed.Call this method again to commit remaining transactions."""
        self.__check_closed()
        assert self.__current_transaction_level > 0

        if self.__current_transaction_level == 1:
            self.__cursor.execute("COMMIT")
        else:
            self.__cursor.execute("RELEASE sp_%d" % (self.__current_transaction_level - 1))
        self.__current_transaction_level -= 1  # in case commit failed

    def rollback(self):
        """Rolls back the transaction. If transaction stack contains more than one transaction, then the only the top-level
        transactions is rolled back.Call this method again to roll back remaining transactions.Note that rolling back a
        transaction also rolls back its nested transactions, that have been already committed."""

        self.__check_closed()
        assert self.__current_transaction_level > 0

        if self.__current_transaction_level == 1:
            self.__cursor.execute("ROLLBACK")
        else:
            self.__cursor.execute("ROLLBACK TO SAVEPOINT sp_%d" % (self.__current_transaction_level - 1))
        self.__current_transaction_level -= 1  # in case rollback failed

    def execute(self, sql: str, parameters=()) -> sqlite3.Cursor:
        return self.__cursor.execute(sql, parameters)

    def executemany(self, sql: str, seq_of_parameters) -> sqlite3.Cursor:
        return self.__cursor.executemany(sql, seq_of_parameters)

    def __check_closed(self):
        if self.__closed:
            raise DatabaseException("Database engine is already closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__check_closed()
        try:
            # DO NOT uncomment: do not commit uncommitted data, user has to explicitly call commit
            # if we close the connection during the transaction, the data is probably corrupt.
            # self.__cursor.execute("COMMIT")

            self.__connection.close()
            # self.__connection.__exit__(exc_type, exc_val, exc_tb)
            self.__closed = True
        except Exception as ex:
            raise DatabaseException("An error occurred during closing database") from ex


class DatabaseException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class Helpers:
    @staticmethod
    def normalize_caseless(text: str):
        return unicodedata.normalize("NFKD", text.casefold())

    @staticmethod
    def equals_ignore_case(left: str, right: str):
        return Helpers.normalize_caseless(left) == Helpers.normalize_caseless(right)

    @staticmethod
    def fix_data_object_name(name: str) -> str:
        # replace invalid characters in column name
        name = re.sub("['`\"\\.]", "_", name)

        if Helpers.equals_ignore_case("id", name) or Helpers.equals_ignore_case("parent", name) or Helpers.equals_ignore_case("sqlite_", name[:7]):
            name = "_" + name

        return name
