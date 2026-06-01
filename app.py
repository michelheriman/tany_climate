"""
Flask API - Rainy Season Onset Analysis
NASA POWER data → JSON

Endpoints:
  GET /api/season?longitude=<float>&latitude=<float>
  GET /api/proximity?longitude=<float>&latitude=<float>

  GET /health
"""

from flask import Flask, jsonify, request
import requests as http
import math
import json
from datetime import date, datetime, timedelta
from statistics import mean, stdev

#from flask import Flask, jsonify, request
from math import radians, sin, cos, sqrt, atan2
import requests
import time

app = Flask(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def calculate_mean(arr):
    return mean(arr)

def stdev_sample(arr):
    return stdev(arr) if len(arr) >= 2 else 0.0

def cumulative_array(arr):
    result, total = [], 0.0
    for v in arr:
        total += v
        result.append(total)
    return result

def get_day_of_year(date_str):
    return datetime.strptime(date_str, "%Y%m%d").timetuple().tm_yday

def get_date_from_doy(year, doy):
    dt = datetime(year, 1, 1) + timedelta(days=int(doy) - 1)
    return {"month": dt.month, "day": dt.day}

def select_trunk(arr, start, length):
    return arr[start: start + length]

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def transform_data(json_data):
    parameters = json_data["properties"]["parameter"]
    all_dates = list(next(iter(parameters.values())).keys())
    rows = []
    for date_key in all_dates:
        row = {"date": date_key}
        for param, values in parameters.items():
            row[param] = values.get(date_key)
        rows.append(row)
    return rows


def climate_stats(rows):
    """
    Compute from all valid daily rows:
      - temperature: overall mean, hottest month (avg T2M_MAX), coldest month (avg T2M_MIN)
      - precipitation: average yearly total, wettest month (avg monthly total)
    """
    # ── temperature ──────────────────────────────────────────────────────────
    # monthly buckets: {month_num: [daily T2M values]}
    monthly_t2m     = {m: [] for m in range(1, 13)}
    monthly_t2m_max = {m: [] for m in range(1, 13)}
    monthly_t2m_min = {m: [] for m in range(1, 13)}
    monthly_prec    = {}   # {(year, month): total_mm}

    for r in rows:
        date_str = r["date"]
        month    = int(date_str[4:6])
        year     = int(date_str[:4])

        t2m     = r.get("T2M")
        t2m_max = r.get("T2M_MAX")
        t2m_min = r.get("T2M_MIN")
        prec    = r.get("PRECTOTCORR")

        if t2m is not None and t2m > -999:
            monthly_t2m[month].append(t2m)
        if t2m_max is not None and t2m_max > -999:
            monthly_t2m_max[month].append(t2m_max)
        if t2m_min is not None and t2m_min > -999:
            monthly_t2m_min[month].append(t2m_min)
        if prec is not None and prec >= 0:
            key = (year, month)
            monthly_prec[key] = monthly_prec.get(key, 0.0) + prec

    # overall mean temperature
    all_t2m = [v for vals in monthly_t2m.values() for v in vals]
    overall_mean_temp = round(mean(all_t2m), 2) if all_t2m else None

    # hottest month = highest mean of T2M_MAX
    month_max_means = {
        m: round(mean(vals), 2)
        for m, vals in monthly_t2m_max.items() if vals
    }
    hottest_month_num  = max(month_max_means, key=month_max_means.get)
    hottest_month_temp = month_max_means[hottest_month_num]

    # coldest month = lowest mean of T2M_MIN
    month_min_means = {
        m: round(mean(vals), 2)
        for m, vals in monthly_t2m_min.items() if vals
    }
    coldest_month_num  = min(month_min_means, key=month_min_means.get)
    coldest_month_temp = month_min_means[coldest_month_num]

    # ── precipitation ─────────────────────────────────────────────────────────
    # group monthly totals by month number across all years
    month_totals = {m: [] for m in range(1, 13)}
    for (year, month), total in monthly_prec.items():
        month_totals[month].append(total)

    # average yearly total = sum of all monthly means
    monthly_means_prec = {
        m: mean(vals) for m, vals in month_totals.items() if vals
    }
    avg_yearly_rain = round(sum(monthly_means_prec.values()), 1)

    # wettest month = highest average monthly total
    wettest_month_num  = max(monthly_means_prec, key=monthly_means_prec.get)
    wettest_month_rain = round(monthly_means_prec[wettest_month_num], 1)

    return {
        "temperature": {
            "overall_mean_c":      overall_mean_temp,
            "hottest_month":       MONTH_NAMES[hottest_month_num],
            "hottest_month_num":   hottest_month_num,
            "hottest_month_avg_max_c": hottest_month_temp,
            "coldest_month":       MONTH_NAMES[coldest_month_num],
            "coldest_month_num":   coldest_month_num,
            "coldest_month_avg_min_c": coldest_month_temp,
        },
        "precipitation": {
            "avg_yearly_total_mm": avg_yearly_rain,
            "wettest_month":       MONTH_NAMES[wettest_month_num],
            "wettest_month_num":   wettest_month_num,
            "wettest_month_avg_mm": wettest_month_rain,
        },
    }


# ── core analysis ─────────────────────────────────────────────────────────────

def season_analysis(longitude, latitude):
    today = date.today()
    end   = today.strftime("%Y%m%d")
    start = "19890101" if latitude > 0 else "19890701"

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?parameters=T2M,T2MDEW,T2MWET,TS,T2M_RANGE,T2M_MAX,T2M_MIN,PRECTOTCORR,EVLAND"
        "&community=RE"
        f"&longitude={longitude}&latitude={latitude}"
        f"&start={start}&end={end}&format=JSON"
    )

    resp = http.get(url, timeout=120)
    resp.raise_for_status()
    raw = resp.json()

    rows      = transform_data(raw)
    valid     = [r for r in rows if r.get("PRECTOTCORR") is not None and r["PRECTOTCORR"] >= 0]
    arr_prec  = [r["PRECTOTCORR"] for r in valid]
    day       = [r["date"]        for r in valid]

    net_year  = round(len(arr_prec) / 365) - 1
    if net_year < 1:
        raise ValueError("Insufficient data to compute rainy season onset.")

    Y, d      = 365, 0
    result_min, results_max, yearly_onsets = [], [], []

    for _ in range(net_year):
        trunked = select_trunk(arr_prec, d, Y)
        if len(trunked) < Y:
            break

        y_mean  = calculate_mean(trunked)
        meaned  = [v - y_mean for v in trunked]
        sd      = stdev_sample(meaned)
        if sd == 0:
            d += Y
            continue

        creduit     = [v / sd for v in meaned]
        result      = cumulative_array(creduit)
        fin_pluie   = result.index(max(result))
        debut_pluie = result.index(min(result))
        day_trunk   = select_trunk(day, d, Y)

        onset_doy   = get_day_of_year(day_trunk[debut_pluie])
        end_doy     = get_day_of_year(day_trunk[fin_pluie])
        result_min.append(onset_doy)
        results_max.append(end_doy)

        year_int    = int(day_trunk[0][:4])
        onset_dt    = datetime(year_int, 1, 1) + timedelta(days=onset_doy - 1)
        end_dt      = datetime(year_int, 1, 1) + timedelta(days=end_doy - 1)
        yearly_onsets.append({
            "year":       year_int,
            "onset_doy":  onset_doy,
            "onset_date": onset_dt.strftime("%Y-%m-%d"),
            "end_doy":    end_doy,
            "end_date":   end_dt.strftime("%Y-%m-%d"),
        })
        d += Y

    if not result_min:
        raise ValueError("Not enough valid yearly slices to compute onset.")

    # onset stats
    mean_onset_doy  = math.floor(calculate_mean(result_min))
    sd_onset_days   = math.floor(stdev_sample(result_min))
    onset_month_day = get_date_from_doy(today.year, mean_onset_doy)

    # end stats
    mean_end_doy    = math.floor(calculate_mean(results_max))
    sd_end_days     = math.floor(stdev_sample(results_max))
    end_month_day   = get_date_from_doy(today.year, mean_end_doy)

    climate = climate_stats(rows)

    return {
        "latitude":    latitude,
        "longitude":   longitude,
        "data_range":  {"start": start, "end": end},
        "rainy_season": {
            "onset": {
                "mean_month":    onset_month_day["month"],
                "mean_day":      onset_month_day["day"],
                "mean_doy":      mean_onset_doy,
                "std_dev_days":  sd_onset_days,
                "interpretation": (
                    f"The rainy season typically starts around "
                    f"{onset_month_day['month']}/{onset_month_day['day']} (month/day), "
                    f"with a standard deviation of {sd_onset_days} days."
                ),
            },
            "end": {
                "mean_month":    end_month_day["month"],
                "mean_day":      end_month_day["day"],
                "mean_doy":      mean_end_doy,
                "std_dev_days":  sd_end_days,
                "interpretation": (
                    f"The rainy season typically ends around "
                    f"{end_month_day['month']}/{end_month_day['day']} (month/day), "
                    f"with a standard deviation of {sd_end_days} days."
                ),
            },
        },
        "temperature":    climate["temperature"],
        "precipitation":  climate["precipitation"],
        #"yearly_onsets":  yearly_onsets,
    }


# ── helpers ────────────────────────────────────────────────────────────────────

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi  = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def query_overpass(query, retries=3):
    for server in OVERPASS_SERVERS:
        for attempt in range(retries):
            try:
                response = requests.get(
                    server,
                    params={"data": query},
                    timeout=30,
                    headers={"User-Agent": "ProximityChecker/1.0"},
                )

                if response.status_code == 429:
                    time.sleep(10)
                    continue

                if response.status_code != 200 or not response.text.strip():
                    break  # try next server

                return response.json()

            except requests.exceptions.Timeout:
                time.sleep(2)
            except requests.exceptions.JSONDecodeError:
                break

    return None


def build_query(lat, lon, radius):
    return f"""
    [out:json][timeout:30];
    (
      node["amenity"="hospital"](around:{radius},{lat},{lon});
      way["amenity"="hospital"](around:{radius},{lat},{lon});
      node["amenity"="clinic"](around:{radius},{lat},{lon});

      node["shop"="supermarket"](around:{radius},{lat},{lon});
      way["shop"="supermarket"](around:{radius},{lat},{lon});
      node["shop"="convenience"](around:{radius},{lat},{lon});

      node["highway"="bus_stop"](around:{radius},{lat},{lon});
      node["railway"="station"](around:{radius},{lat},{lon});
      node["railway"="tram_stop"](around:{radius},{lat},{lon});
      node["public_transport"="station"](around:{radius},{lat},{lon});
    );
    out center;
    """


def parse_elements(elements, lat, lon):
    results = {"hospital": [], "market": [], "transport": []}

    for el in elements:
        tags    = el.get("tags", {})
        name    = tags.get("name", "Unnamed")
        el_lat  = el.get("lat") or el.get("center", {}).get("lat")
        el_lon  = el.get("lon") or el.get("center", {}).get("lon")

        if not el_lat:
            continue

        distance   = round(haversine(lat, lon, el_lat, el_lon))
        amenity    = tags.get("amenity", "")
        shop       = tags.get("shop", "")
        highway    = tags.get("highway", "")
        railway    = tags.get("railway", "")
        public_tr  = tags.get("public_transport", "")

        entry = {
            "name":       name,
            "distance_m": distance,
            "lat":        el_lat,
            "lon":        el_lon,
        }

        if amenity in ("hospital", "clinic"):
            results["hospital"].append(entry)
        elif shop in ("supermarket", "convenience"):
            results["market"].append(entry)
        elif highway == "bus_stop" or railway in ("station", "tram_stop") or public_tr == "station":
            results["transport"].append(entry)

    for key in results:
        results[key].sort(key=lambda x: x["distance_m"])

    return results



# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/season")
def season():
    # --- validate query params ---
    errors = {}
    try:
        longitude = float(request.args["longitude"])
    except (KeyError, ValueError):
        errors["longitude"] = "Required float query parameter."

    try:
        latitude = float(request.args["latitude"])
    except (KeyError, ValueError):
        errors["latitude"] = "Required float query parameter."

    if errors:
        return jsonify({"error": "Invalid parameters", "details": errors}), 400

    # --- run analysis ---
    try:
        result = season_analysis(longitude, latitude)
        return jsonify(result)
    except http.exceptions.Timeout:
        return jsonify({"error": "NASA POWER API timed out. Try again later."}), 504
    except http.exceptions.HTTPError as e:
        return jsonify({"error": f"NASA POWER API error: {str(e)}"}), 502
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": "Unexpected error", "details": str(e)}), 500

@app.route("/api/proximity", methods=["GET"])
def proximity():
    # --- validate inputs ---
    lat_str = request.args.get("latitude")
    lon_str = request.args.get("longitude")
    radius  = request.args.get("radius", 1000)  # optional, default 1 km

    if not lat_str or not lon_str:
        return jsonify({"error": "Missing required parameters: lat and lon"}), 400

    try:
        lat    = float(lat_str)
        lon    = float(lon_str)
        radius = int(radius)
    except ValueError:
        return jsonify({"error": "lat, lon must be floats and radius must be an integer"}), 400

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return jsonify({"error": "lat must be between -90 and 90, lon between -180 and 180"}), 400

    if not (100 <= radius <= 10000):
        return jsonify({"error": "radius must be between 100 and 10000 meters"}), 400

    # --- query ---
    data = query_overpass(build_query(lat, lon, radius))

    if data is None:
        return jsonify({"error": "Overpass API unavailable. Try again later."}), 503

    results = parse_elements(data.get("elements", []), lat, lon)

    return jsonify({
        "query": {"lat": lat, "lon": lon, "radius_m": radius},
        "results": results,
    })


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)