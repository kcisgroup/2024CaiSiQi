from common import register_api
from .match import MatchAPI


def register_match_api(api_bp):
    register_api(MatchAPI, 'match_api', '/match/match/', pk='match_id', app=api_bp)