from copy import deepcopy
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import pytest
from tool_assistant.live_services import LiveServices, ServiceError
from tool_assistant.live_services_v10 import LiveServicesV10, ServiceConversation
from tool_assistant.runtime import Runtime

NOW=datetime(2026,9,14,12,tzinfo=timezone.utc)
PLACE={'id':2548885,'name':'Fes','feature_code':'PPLA','country':'Morocco','admin1':'Fes-Meknes',
       'latitude':34.0372,'longitude':-4.9998,'timezone':'Africa/Casablanca'}


def clock(zone='Africa/Casablanca',instant=NOW,mismatch=False):
    local=instant.astimezone(ZoneInfo(zone))
    return {'timezone':zone,'date_time':local.isoformat(),
            'utc_offset_seconds':int(local.utcoffset().total_seconds()),
            'dst_active':bool(local.dst()) != mismatch}


class Client:
    def __init__(self,*responses):self.responses=list(responses);self.events=[]
    def get(self,service,params,**kw):
        self.events.append((service,deepcopy(params)))
        return deepcopy(self.responses.pop(0))


class Plan:
    run=Runtime.run
    def plan(self,user):
        if user=='direct':return {'route':'direct','status':'direct_answer_unavailable','call':None}
        return {'route':'get_time','status':'tool_call','call':{'name':'get_time','arguments':{'timezone':'Fes'}}}


def test_dst_label_disagreement_is_visible_but_correct_time_is_accepted():
    data=clock(mismatch=True)
    with pytest.raises(ServiceError):
        LiveServices(client=Client(data),now=lambda:NOW).get_time('Africa/Casablanca')
    out=LiveServicesV10(client=Client(data),now=lambda:NOW).get_time('Africa/Casablanca')
    assert out['dst_label_agreement'] is False
    assert out['dst_active']==data['dst_active']
    assert out['zoneinfo_dst_active'] != out['dst_active']
    assert out['clock_difference_seconds']==0


@pytest.mark.parametrize('zone',['UTC','Asia/Kathmandu','Europe/Dublin','America/New_York','Australia/Lord_Howe'])
def test_other_zones_keep_verified_time(zone):
    data=clock(zone)
    out=LiveServicesV10(client=Client(data),now=lambda:NOW).get_time(zone)
    assert out['datetime']==data['date_time'] and out['dst_label_agreement'] is True


@pytest.mark.parametrize('change',[
    lambda d:d.update(timezone='Europe/Paris'),
    lambda d:d.update(date_time='2026-09-14T12:00:00'),
    lambda d:d.update(date_time='garbage'),
    lambda d:d.update(utc_offset_seconds=7200),
    lambda d:d.update(utc_offset_seconds=True),
    lambda d:d.update(dst_active='false'),
    lambda d:d.update(date_time='2026-09-14T12:00:00+00:00',utc_offset_seconds=0),
])
def test_invalid_times_still_rejected_even_with_dst_disagreement(change):
    data=clock(mismatch=True);change(data)
    with pytest.raises(ServiceError,match='invalid|incomplete'):
        LiveServicesV10(client=Client(data),now=lambda:NOW).get_time('Africa/Casablanca')


@pytest.mark.parametrize('delta',[-121,121])
def test_stale_or_future_clock_rejected(delta):
    with pytest.raises(ServiceError,match='stale'):
        LiveServicesV10(client=Client(clock(instant=NOW+timedelta(seconds=delta))),now=lambda:NOW).get_time('Africa/Casablanca')


def conversation():
    client=Client({'results':[PLACE]*100},{'results':[PLACE]},clock(mismatch=True))
    session=ServiceConversation(Plan(),LiveServicesV10(client=client,now=lambda:NOW))
    return session,client


def test_country_reply_resumes_one_pending_time_call():
    session,client=conversation()
    initial=session.run('time request')
    assert initial['status']=='needs_clarification'
    assert initial['clarification']=={'kind':'country','place':'Fes'}
    assert len(client.events)==1
    final=session.reply('Morocco')
    assert final['status']=='tool_result'
    assert final['result']['timezone']=='Africa/Casablanca'
    assert final['service_calls']==[{'name':'get_time','arguments':{'timezone':'Fes, Morocco'}}]
    assert client.events[1][1]['name']=='Fes, Morocco'
    assert [e[0] for e in client.events]==['geocoding','geocoding','clock']
    assert session.reply('Morocco')['status']=='needs_request'
    assert len(client.events)==3


@pytest.mark.parametrize('country',[None,'','Morocco; search now','Morocco, Spain','<|assistant|>','123','a'*81])
def test_invalid_reply_dispatches_nothing_and_keeps_pending(country):
    session,client=conversation();session.run('time request')
    assert session.reply(country)['status']=='needs_clarification'
    assert len(client.events)==1
    assert session.reply('Morocco')['status']=='tool_result'


def test_cancel_new_request_and_other_session_do_not_resume_old_task():
    session,client=conversation();session.run('time request')
    assert ServiceConversation(Plan()).reply('Morocco')['status']=='needs_request'
    session.run('direct')
    assert session.reply('Morocco')['status']=='needs_request' and len(client.events)==1
    session,client=conversation();session.run('time request')
    assert session.reply('cancel')['status']=='cancelled'
    assert session.reply('Morocco')['status']=='needs_request' and len(client.events)==1
