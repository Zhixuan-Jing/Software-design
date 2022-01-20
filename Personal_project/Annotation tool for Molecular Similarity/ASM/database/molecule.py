from ASM import db
from ASM.models import Molecule
def batch(number, size = 5):
  entity = db.session.query(Molecule).all()
  size = len(entity)
  answerset = {}
  cursor = []
  for i in min(size,range(number)):
    if i%6 == 0:
      answerset[entity[i].id]= [] 
      cursor = answerset[entity[i].id]
    else:
      cursor.append(entity[i])
  return answerset


