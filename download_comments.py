import praw

def expand(comment):
    out = [comment]
    for reply in comment.replies:
        out += expand(reply)
    return out

reddit = praw.Reddit(client_id='PTofuEjEjIPbcg',
                     client_secret='_R0b3zmCvjXGPseYbaPIUEnZAlU',
                     password='LinguisticsIsCool208',
                     user_agent='testscript by /u/conor_emily_ling208',
                     username='conor_emily_ling208')
reddit.read_only = True
rscotland = reddit.subreddit('Scotland')

all_comments = []
for submission in rscotland.top(limit=25):
    comment_forest = submission.comments
    comment_forest.replace_more()
    for comment in list(comment_forest):
        all_comments += expand(comment)

comment_bodies = [comment.body.replace('\u200b','') for comment in all_comments]
#if we want to ignore newlines in our language model:
comment_bodies = [comment_body.replace('\n',' ') for comment_body in comment_bodies]
with open('rscotland_corpus.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(comment_bodies))
