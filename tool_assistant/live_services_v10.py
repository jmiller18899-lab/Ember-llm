"""Optional clock validation and explicit country-clarification continuation."""
from copy import deepcopy
from datetime import datetime
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from .live_services import LiveServices, ServiceError, require
from .runtime import Runtime

REVISION = 'live-services-v10'


class LiveServicesV10(LiveServices):
    def get_time(self, timezone):
        location = None
        try:
            zone = ZoneInfo(timezone)
        except (ValueError, ZoneInfoNotFoundError):
            if '/' in timezone:
                raise ServiceError('invalid_timezone', 'Please give a valid IANA time zone or a city and country.') from None
            location = self.place(timezone)
            zone = ZoneInfo(location['timezone'])
        data = self.client.get('clock', {'timezone': zone.key})
        require(data.get('timezone') == zone.key and isinstance(data.get('date_time'), str))
        require(type(data.get('dst_active')) is bool)
        require(type(data.get('utc_offset_seconds')) is int)
        try:
            local = datetime.fromisoformat(data['date_time'])
            require(local.tzinfo is not None)
            expected = local.astimezone(zone)
            require(local.utcoffset() == expected.utcoffset())
            require(data['utc_offset_seconds'] == expected.utcoffset().total_seconds())
        except (ValueError, OverflowError):
            raise ServiceError('invalid_response', 'The clock service returned an invalid timestamp.') from None
        now = self.now()
        age = now.timestamp() - local.timestamp()
        if abs(age) > 120:
            raise ServiceError('stale_response', 'The clock service returned a stale or incorrect time.')
        # The zone, timestamp offset, numeric offset, and freshness must agree.
        # DST labels are metadata: providers/tzdb versions can classify the same
        # UTC offset differently. Preserve both labels rather than invent agreement.
        zone_dst = bool(expected.dst())
        return {'provider': 'TimeAPI.io', 'attribution_url': 'https://timeapi.io/',
                'timezone': zone.key, 'datetime': local.isoformat(), 'location': location,
                'utc_offset_seconds': data['utc_offset_seconds'],
                'dst_active': data['dst_active'], 'dst_active_source': 'provider',
                'zoneinfo_dst_active': zone_dst,
                'dst_label_agreement': data['dst_active'] == zone_dst,
                'fetched_at': now.isoformat(), 'clock_difference_seconds': round(age, 3),
                'service_revision': REVISION}


class _ClarifiedCall:
    run = Runtime.run

    def __init__(self, call):
        self.call = deepcopy(call)

    def plan(self, user):
        return {'route': self.call['name'], 'status': 'tool_call',
                'call': deepcopy(self.call), 'routing_source': 'explicit_country_clarification',
                'service_revision': REVISION}


class ServiceConversation:
    """One instance per conversation. run starts a request; reply supplies country.

    No country is inferred from provider suggestions. State is local to this
    instance; a new request or successful reply clears the pending request.
    """
    def __init__(self, runtime, services=None):
        self.runtime = runtime
        self.services = services if services is not None else LiveServicesV10()
        self._pending = None

    def _remember(self, result):
        call = result.get('call') or {}
        code = result.get('service_error', {}).get('code')
        if call.get('name') in ('weather', 'get_time') and code in ('ambiguous_location', 'location_not_found'):
            key = 'location' if call['name'] == 'weather' else 'timezone'
            place = call['arguments'][key].split(',')[0].strip()
            self._pending = (deepcopy(call), key, place)
            return {**result, 'clarification': {'kind': 'country', 'place': place},
                    'message': f'Which country is {place} in? Please reply with the country name.'}
        return result

    def run(self, user):
        self._pending = None
        return self._remember(self.services.run(self.runtime, user))

    def reply(self, country):
        if self._pending is None:
            return {'status': 'needs_request', 'call': None, 'service_calls': [],
                    'message': 'Please start a weather or time request first.'}
        if isinstance(country, str) and country.strip().casefold() == 'cancel':
            self._pending = None
            return {'status': 'cancelled', 'call': None, 'service_calls': []}
        country = country.strip() if isinstance(country, str) else ''
        if not country or len(country) > 80 or not re.fullmatch(r"[^\W\d_]+(?:[ '\u2019-][^\W\d_]+)*", country):
            return {'status': 'needs_clarification', 'call': None, 'service_calls': [],
                    'message': 'Please reply with only the country name, or cancel.'}
        call, key, place = self._pending
        self._pending = None
        call = deepcopy(call)
        call['arguments'][key] = f'{place}, {country}'
        result = self.services.run(_ClarifiedCall(call), country)
        return self._remember({**result, 'clarified_location': call['arguments'][key]})
