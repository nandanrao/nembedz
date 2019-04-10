
# with open('../text-mining/trolls/russia_201901_1_tweets_csv_hashed.csv') as f:
#     ds = enumerate(csv.DictReader(f))
#     users = ((i, [d['userid']])
#              for i,d in ds)
#     user_lookup = Lookup()
#     user_lookup.add_to_lookups(users)

# with open('../text-mining/trolls/russia_201901_1_tweets_csv_hashed.csv') as f:
#     ds = enumerate(csv.DictReader(f))
#     tags = ((i, parse_tags(d['hashtags']))
#             for i,d in ds)
#     hashtag_lookup = Lookup()
#     hashtag_lookup.add_to_lookups(tags)


tag_pat = re.compile(r'\[|\]')

def parse_tags(s):
    s = re.sub(tag_pat, '', s)
    return [x.strip() for x in s.split(',')]


class Lookup():
    def __init__(self):
        self.rl = {}
        self.fl = {}

    def _add_to_lookup(self, lookup, i, v):
        try:
            lookup[i] += [v]
        except KeyError:
            lookup[i] = [v]

    def add_to_lookups(self, items:Sequence[Tuple[int,Sequence[str]]]):
        for i,values in items:
            for v in values:
                self._add_to_lookup(self.fl, i, v)
                self._add_to_lookup(self.rl, v, i)

    def get_ids(self, val):
        return self.rl[val]

    def get_vals(self, i):
        return self.fl[i]
