from google.cloud import bigquery

from config import GOOGLE_APPLICATION_CREDENTIALS, POST_TABLE, COMMENT_TABLE


class Querier(object):
    def __init__(self):
        self.post_table_name = POST_TABLE
        self.comment_table_name = COMMENT_TABLE
        self.client = bigquery.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)

    def get_posts(self):
        query_job = self.client.query(
            """SELECT * FROM %s
            where num_comments	>= 5 """ % self.post_table_name)

        return query_job

    def get_comments(self, post_id):
        query_job = self.client.query(
            """SELECT * FROM %s
            where SUBSTR(link_id, 4) = '%s' and score >= 0""" % (self.comment_table_name, post_id))

        return query_job

    def get_post_data(self):
        post_table = self.get_posts().to_dataframe()
        for i in range(len(post_table)):
            row = post_table.iloc[i]
            print(row.title, row.id, type(row.id))
            query_job = self.get_comments(row.id)
            print(query_job.query)
            results = query_job.result()
            comments = [{'cmt_utc': x.created_utc, 'cmt_text': x.body, 'cmt_score': x.score} for x in results]
            yield {'utc': row.created_utc,
                   'title': row.title,
                   'text': row.selftext,
                   'score': row.score,
                   'comments': comments}
