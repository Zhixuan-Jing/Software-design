from ASM import db

class Molecule(db.Model):
  __tablename__="molecule"
  
  id = db.Column(db.Integer, primary_key=True) 
  picture = db.Column(db.String) 
  smiles = db.Column(db.String)

  def to_json(self):
    dict = self.__dict__
    if '_sa_instance_state' in dict:
        del dict['_sa_instance_state']
    return dict

class Similarity(db.Model):
  __tablename__="similarity"

  target = db.Column(db.Integer)
  query = db.Column(db.Integer)
  score = db.Column(db.Integer)

  def to_json(self):
    dict = self.__dict__
    if '_sa_instance_state' in dict:
        del dict['_sa_instance_state']
    return dict