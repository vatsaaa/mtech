import json

import boto3
from boto3.dynamodb.conditions import Key

class InvalidResponse(Exception):
	def __init__(self, status_code):
            self.status_code = status_code

def query_user_notes(user_email):
    dynamo_db = boto3.resource('dynamodb')
    user_notes_table = dynamo_db.Table('user-notes')
    result = user_notes_table.query(
         KeyConditionExpression=Key('user').eq(user_email)
    )
    
    return result['Items']

def get_authenticated_user_email(token):
    dynamo_db =  boto3.resource('dynamodb')
    tokens_table = dynamo_db.Table('token-email-lookup')
    result = tokens_table.get_item(Key={'token': token})

    item = result.get('Item')
    return item.get('email') if item else None
    
def authenticate_user(headers):
    authentication_header = headers.get('Authentication')
    
    if not authentication_header or not authentication_header.startswith('Bearer '):
        raise InvalidResponse(status_code=400)
    
    token = authentication_header.split(' ')[1]
    user_email = get_authenticated_user_email(token)
    
    if not user_email:
        raise InvalidResponse(status_code=403)

    return user_email

def build_response(status_code, body=None):
    result = {
        'statusCode': str(status_code),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }
    if body is not None:
        result['body'] = body
    
    return result

def handler(event: dict, context):
    try:
        user_email = authenticate_user(event['headers'])
        notes = query_user_notes(user_email)
        return build_response(status_code=200, body=json.dumps(notes))
    except InvalidResponse as e:
        return build_response(status_code=e.status_code)

