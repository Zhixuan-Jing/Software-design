from ASM import db
from ASM.models import Similarity

def score(target,query,score):
  score = Similarity(target, query, score)
  db.session.add(score)
  db.session.commit()
  return True