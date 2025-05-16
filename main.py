# import sys, os
# sys.path.insert(
#     0, os.path.abspath("../")
# )  # Adds the parent directory to the system path
import os 
import secrets
import traceback
import llm as llm
from utils import getenv, set_env_variables
import json, time

import litellm
from litellm import BudgetManager
from litellm.caching import Cache
litellm.max_budget = 1000 

budget_manager = BudgetManager(project_name=os.getenv("PROJECT_NAME"), client_type="hosted")

from fastapi import FastAPI, Request, status, HTTPException, Depends, Body
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
master_key = os.getenv("LITELLM_PROXY_MASTER_KEY", "sk-litellm-master-key")

# Initialize Redis cache if Redis is available
user_api_key_cache = None
redis_host = os.getenv("REDISHOST")
redis_port = os.getenv("REDISPORT")
redis_password = os.getenv("REDISPASSWORD")

if redis_host and redis_port:
    try:
        user_api_key_cache = Cache(
            type="redis",
            host=redis_host,
            port=redis_port,
            password=redis_password if redis_password else None,
            namespace="litellm:user_api_keys:"
        )
        print("Redis cache for user API keys initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Redis cache: {str(e)}")

# Load user API keys from budget manager
user_api_keys = set(budget_manager.get_users())
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

######## AUTH UTILITIES ################

def user_api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key == master_key:
        return
    
    # Check in-memory cache first
    if api_key in user_api_keys:
        return
    
    # If Redis cache is available, check there too
    if user_api_key_cache:
        try:
            # Try to get the user from Redis cache
            cached_user = user_api_key_cache.get_cache(key=f"user:{api_key}")
            if cached_user:
                # Add to in-memory cache for faster future lookups
                user_api_keys.add(api_key)
                return
        except Exception as e:
            print(f"Error checking Redis cache: {str(e)}")
    
    # If we get here, the key is not valid
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "invalid user key"},
        # TODO: this will be {'detail': {'error': 'something'}}
    )


def key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key != master_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid admin key"},
            # TODO: this will be {'detail': {'error': 'something'}}
        )

######## CHAT COMPLETIONS ################

# for streaming
def data_generator(response):
    for chunk in response:
        # print(f"chunk: {chunk}")
        yield f"data: {json.dumps(chunk)}\n\n"

# for completion
@app.post("/chat/completions", dependencies=[Depends(user_api_key_auth)])
async def completion(request: Request):
    key = request.headers.get("Authorization").replace("Bearer ", "")  # type: ignore
    data = await request.json()
    print(f"received request data: {data}")
    data["user_key"] = key
    data["budget_manager"] = budget_manager
    data["master_key"] = master_key
    set_env_variables(data)
    # handle how users send streaming
    if 'stream' in data:
        if type(data['stream']) == str: # if users send stream as str convert to bool
            # convert to bool
            if data['stream'].lower() == "true":
                data['stream'] = True # convert to boolean
    
    response = llm.completion(**data)
    if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return StreamingResponse(data_generator(response), media_type='text/event-stream')
    return response


@app.get("/models/available")
def get_available_models():
    return {"models": litellm.utils.get_valid_models()}


@app.get("/models") # if project requires model list 
def model_list(): 
    available_models = litellm.utils.get_valid_models()
    data = []
    for model in available_models: 
        {
            "id": model, 
            "object": model, 
            "created": int(time.time()), 
            "owned_by": "openai"
        }
    return dict(
        data=data,
        object="list",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}

######## KEY MANAGEMENT ################

@app.get("/key/cost", dependencies=[Depends(user_api_key_auth)])
async def report_current(request: Request):
    key = request.headers.get("Authorization").replace("Bearer ", "")  # type: ignore
    return budget_manager.get_model_cost(key)


class UserUpdateWithIdRequest(BaseModel):
    user_id: str
    total_budget: float = None
    duration: str = None
    metadata: dict = None

@app.post("/user/update", dependencies=[Depends(key_auth)])
async def update_user(user_data: UserUpdateWithIdRequest):
    try:
        user_id = user_data.user_id
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "user_id is required"}
            )
        
        # Update user in budget manager
        update_data = {}
        if user_data.total_budget is not None:
            update_data["total_budget"] = user_data.total_budget
        if user_data.duration is not None:
            update_data["duration"] = user_data.duration
        if user_data.metadata is not None:
            update_data["metadata"] = user_data.metadata
            
        # Update user in budget manager
        budget_manager.update_budget(user=user_id, **update_data)
        
        # Update user in Redis cache if available
        if user_api_key_cache:
            try:
                # Get current user data
                user_data = budget_manager.get_budget(user=user_id)
                # Update in Redis cache
                user_api_key_cache.add_cache(
                    result=user_data,
                    key=f"user:{user_id}"
                )
                print(f"Updated user {user_id} in Redis cache")
            except Exception as e:
                print(f"Error updating Redis cache: {str(e)}")
        
        return {"status": "success", "user_id": user_id, "updated": update_data}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e)}
        )

@app.post("/key/new", dependencies=[Depends(key_auth)])
async def generate_key(request: Request):
    try:
        data = await request.json()
        data.get("total_budget")
    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

    total_budget = data["total_budget"]

    api_key = f"sk-litellm-{secrets.token_urlsafe(16)}"

    try:
        budget_manager.create_budget(
            total_budget=total_budget, user=api_key, duration="monthly"
        )
        user_api_keys.add(api_key)
        
        # Add to Redis cache if available
        if user_api_key_cache:
            try:
                user_data = budget_manager.get_budget(user=api_key)
                user_api_key_cache.add_cache(
                    result=user_data,
                    key=f"user:{api_key}"
                )
                print(f"Added new user {api_key} to Redis cache")
            except Exception as e:
                print(f"Error adding to Redis cache: {str(e)}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return {"api_key": api_key, "total_budget": total_budget, "duration": "monthly"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=getenv("PORT", 8080))
