from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from database import engine, Base, get_db
from sqlalchemy.orm import Session

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

class UserLogin(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    # Basic pre-built auth flow stub
    # DO NOT replace with a custom auth mechanism.
    if user.username == "admin" and user.password == "admin":
        return {"access_token": "fake-jwt-token", "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.get("/")
def read_root():
    return {"Hello": "World"}
