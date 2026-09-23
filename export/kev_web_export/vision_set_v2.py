"""Build eval/vision-v2: harder image decision questions, to find where Kev + the stock vision tower breaks.

    uv run --group vision python -m kev_web_export.vision_set_v2 --out ../eval/vision-v2
    node scripts/render-vision-set.mjs eval/vision-v2          # screenshots pages/*.html -> images/*.png (Chromium)

Four kinds of difficulty, all generated from a seeded RNG so every answer is known by construction:

- screens: app pages at real screen sizes (1280 x 800 and up), rendered by Chromium from pages/*.html: an inbox, a shop
  grid, a KPI dashboard, an orders table. Answers need reading one row among a dozen, or counting rows by state.
- charts: dense SVG charts, a 12-month line chart and a 10-bar chart, where answers need reading values off the axis
  or comparing bars a few percent apart.
- counting: 8 to 30 dots drawn with Pillow; count all of them, or only one color.
- small text: a flight departures board, a shipping label and terms fine print at 11-13 px, where the answer is in
  a line of small print.

Images above the 589,824-pixel cap (kev_web_export.vision.MAX_PIXELS) are downscaled by Qwen's processor like any
other input, which is part of what this set measures. Each item has a neutral `context` (shown in every condition)
and a `caption` with everything the questions need (the text-only upper bound), as in vision-v1."""
import argparse, json, os, random
from PIL import Image, ImageDraw
from .vision_set import noul, choice, score

CSS = """<style>
* { box-sizing: border-box; } body { margin: 0; font-family: Helvetica, Arial, sans-serif; color: #1f2328; background: #f6f8fa; }
.bar { background: #24292f; color: #fff; padding: 14px 24px; font-size: 18px; } .wrap { padding: 20px 24px; }
table { border-collapse: collapse; width: 100%; background: #fff; } td, th { border-bottom: 1px solid #d0d7de; padding: 9px 12px; text-align: left; font-size: 15px; }
th { background: #f0f3f6; font-size: 13px; text-transform: uppercase; color: #57606a; }
.card { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; }
.grid { display: grid; gap: 16px; } .muted { color: #57606a; } .small { font-size: 12px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; }
</style>"""


def page(title, body, width=1280, height=800):
    return {"html": f"<!doctype html><html><head><meta charset='utf-8'>{CSS}</head><body style='width:{width}px;height:{height}px;overflow:hidden'>"
                    f"<div class='bar'>{title}</div><div class='wrap'>{body}</div></body></html>", "size": [width, height]}


SENDERS = ["Maria Lopez", "GitHub", "Tom Becker", "Stripe", "Priya Nair", "Airline Desk", "Jonas Weber", "HR Team", "Linear", "Ana Costa", "Bank Alerts", "Chen Wei"]
SUBJECTS = ["Quarterly report draft", "Your invoice is ready", "Lunch on Friday?", "Payment received", "Design review notes", "Flight change confirmed",
            "Server migration plan", "Benefits enrollment", "Weekly digest", "Contract renewal", "Unusual sign-in attempt", "Team offsite agenda"]


def inbox(rng, i):
    n = 12
    order = rng.sample(range(12), n)
    unread = sorted(rng.sample(range(n), [2, 4, 5, 7][i]))
    rows = "".join(f"<tr style='font-weight:{700 if k in unread else 400}'><td>{'●' if k in unread else ''}</td><td>{SENDERS[s]}</td><td>{SUBJECTS[s]}</td>"
                   f"<td class='muted'>{9 + k // 2}:{(k * 7) % 60:02d} {'AM' if 9 + k // 2 < 12 else 'PM'}</td></tr>" for k, s in enumerate(order))
    html = page("Mail - Inbox", f"<table><tr><th></th><th>From</th><th>Subject</th><th>Received</th></tr>{rows}</table>")
    top = SENDERS[order[0]]
    # every sender appears once with its own subject, so asking for a sender with another row's subject is a "no"
    present = i % 2 == 0
    s_a, s_b = order[3], order[5]
    subj = SUBJECTS[s_a] if present else SUBJECTS[s_b]
    cands = [top] + rng.sample([SENDERS[s] for s in order[1:]], 3); rng.shuffle(cands)
    cap = "An email inbox, newest first: " + "; ".join(f"{SENDERS[s]} - {SUBJECTS[s]} ({'unread' if k in unread else 'read'})" for k, s in enumerate(order)) + "."
    return html, "A screenshot of an email inbox.", cap, {
        "unread": score("How many emails are unread?", [str(k) for k in range(9)], len(unread)),
        "pair": noul(f"Is there an email from {SENDERS[s_a]} with the subject '{subj}'?", present),
        "top": choice("Who sent the email at the top of the list?", cands, top),
    }


PRODUCTS = ["Desk lamp", "Water bottle", "Backpack", "Headphones", "Notebook set", "Coffee grinder", "Yoga mat", "Phone stand",
            "Wool socks", "Travel mug", "Keyboard", "Plant pot"]


def shop(rng, i):
    items = rng.sample(PRODUCTS, 8)
    while True:
        prices = [rng.choice([9, 14, 19, 24, 29, 39, 45, 49, 59, 64, 79, 89, 119]) + 0.99 for _ in items]
        if prices.count(min(prices)) == 1: break
    out = set(rng.sample(range(8), 2))
    cards = "".join(f"<div class='card'><div style='height:120px;background:#e6ebf1;border-radius:6px;margin-bottom:10px'></div><div style='font-size:16px'>{n}</div>"
                    f"<div style='font-size:18px;font-weight:700;margin:4px 0'>${p:.2f}</div>"
                    f"<div class='small' style='color:{'#cf222e' if k in out else '#1a7f37'}'>{'Out of stock' if k in out else 'In stock'}</div></div>"
                    for k, (n, p) in enumerate(zip(items, prices)))
    html = page("Shop - Home office", f"<div class='grid' style='grid-template-columns:repeat(4,1fr)'>{cards}</div>", 1280, 900)
    cheap = items[prices.index(min(prices))]
    k = rng.randrange(8); ask_in = k not in out
    cands = [cheap] + rng.sample([x for x in items if x != cheap], 3); rng.shuffle(cands)
    over = sum(p > 50 for p in prices)
    cap = "An online shop grid: " + "; ".join(f"{n} ${p:.2f} ({'out of stock' if j in out else 'in stock'})" for j, (n, p) in enumerate(zip(items, prices))) + "."
    return html, "A screenshot of an online shop.", cap, {
        "cheap": choice("Which of these products is the cheapest?", cands, cheap),
        "stock": noul(f"Is the {items[k].lower()} in stock?", ask_in),
        "over50": score("How many products cost more than $50?", [str(j) for j in range(9)], over),
    }


def dashboard(rng, i):
    up = i % 2 == 0
    rev, rchg = rng.randint(10, 90) * 1000 + rng.randint(0, 999), rng.choice([1.8, 3.4, 5.1, 7.9]) * (1 if up else -1)
    churn = [1.2, 2.7, 4.6, 6.3][i]
    users, uchg = rng.randint(2000, 9000), rng.choice([2.2, 4.4, 6.1]) * (-1 if up else 1)
    tickets = rng.randint(20, 90)
    arrow = lambda c: f"<span style='color:{'#1a7f37' if c > 0 else '#cf222e'}'>{'▲' if c > 0 else '▼'} {abs(c):.1f}%</span>"
    cards = [("Revenue (MTD)", f"${rev:,}", arrow(rchg)), ("Monthly churn", f"{churn:.1f}%", "<span class='muted'>vs 3.0% target</span>"),
             ("Active users", f"{users:,}", arrow(uchg)), ("Open tickets", str(tickets), "<span class='muted'>support queue</span>")]
    body = "<div class='grid' style='grid-template-columns:repeat(4,1fr)'>" + "".join(
        f"<div class='card'><div class='small muted'>{t}</div><div style='font-size:28px;font-weight:700;margin:6px 0'>{v}</div><div class='small'>{d}</div></div>" for t, v, d in cards) + "</div>"
    body += "<div class='card' style='margin-top:16px;height:420px'><div class='small muted'>Sessions per day</div></div>"
    html = page("Analytics - Overview", body)
    cap = (f"An analytics dashboard: revenue month to date ${rev:,}, {'up' if rchg > 0 else 'down'} {abs(rchg):.1f}%; monthly churn {churn:.1f}% vs a 3.0% target; "
           f"active users {users:,}, {'up' if uchg > 0 else 'down'} {abs(uchg):.1f}%; {tickets} open tickets.")
    return html, "A screenshot of an analytics dashboard.", cap, {
        "revenue": noul("Did revenue go up?", rchg > 0),
        "churn": score("How high is monthly churn?", ["under 2%", "2% to 4%", "4% to 6%", "over 6%"], [0, 1, 2, 3][i]),
        "down": choice("Which metric went down?", ["revenue", "active users"], "revenue" if rchg < 0 else "active users"),
        "target": noul("Is churn above its target?", churn > 3.0),
    }


CUSTOMERS = ["Acme Corp", "Blue Fin", "Cedar LLC", "Delta Foods", "Evergreen", "Fox & Co", "Granite", "Harbor Labs", "Iris Media", "Juniper"]
STATUS = {"Shipped": "#1a7f37", "Pending": "#9a6700", "Refunded": "#cf222e", "Delivered": "#0969da"}


def orders(rng, i):
    n = 10
    custs = rng.sample(CUSTOMERS, n)
    while True:
        totals = [rng.randint(40, 990) + rng.choice([0, 0.5, 0.99]) for _ in custs]
        if sorted(totals)[-1] - sorted(totals)[-2] > 30: break
    refunded = [2, 3, 1, 4][i]
    st = ["Refunded"] * refunded + [rng.choice(["Shipped", "Pending", "Delivered"]) for _ in range(n - refunded)]; rng.shuffle(st)
    ids = [1040 + k for k in range(n)]
    rows = "".join(f"<tr><td>#{o}</td><td>{c}</td><td>${t:,.2f}</td><td><span class='badge' style='background:{STATUS[s]}22;color:{STATUS[s]}'>{s}</span></td></tr>"
                   for o, c, t, s in zip(ids, custs, totals, st))
    html = page("Admin - Orders", f"<table><tr><th>Order</th><th>Customer</th><th>Total</th><th>Status</th></tr>{rows}</table>")
    k = rng.randrange(n)
    biggest = custs[totals.index(max(totals))]
    cands = [biggest] + rng.sample([c for c in custs if c != biggest], 3); rng.shuffle(cands)
    cap = "An orders table: " + "; ".join(f"order #{o}, {c}, ${t:,.2f}, {s}" for o, c, t, s in zip(ids, custs, totals, st)) + "."
    return html, "A screenshot of an admin orders page.", cap, {
        "refunded": score("How many orders are refunded?", [str(j) for j in range(7)], refunded),
        "shipped": noul(f"Is order #{ids[k]} shipped?", st[k] == "Shipped"),
        "biggest": choice("Which customer has the largest order total?", cands, biggest),
    }


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def svg_page(title, svg, w=960, h=640):
    return {"html": f"<!doctype html><html><head><meta charset='utf-8'>{CSS}</head><body style='width:{w}px;height:{h}px;background:#fff;overflow:hidden'>"
                    f"<div style='padding:16px 24px;font-size:20px'>{title}</div>{svg}</body></html>", "size": [w, h]}


def line_chart(rng, i):
    while True:
        vals = [rng.randint(12, 88) for _ in MONTHS]
        top = max(vals)
        if vals.count(top) == 1 and sorted(vals)[-2] <= top - 4 and abs(vals[5] - 50) >= 4 and vals[2] % 10 not in (0, 9, 1): break
    W, H, x0, y0, x1, y1 = 960, 560, 70, 20, 930, 520
    X = lambda k: x0 + (x1 - x0) * k / 11
    Y = lambda v: y1 - (y1 - y0) * v / 100
    g = "".join(f"<line x1='{x0}' x2='{x1}' y1='{Y(t)}' y2='{Y(t)}' stroke='#e1e4e8'/><text x='{x0 - 10}' y='{Y(t) + 4}' font-size='12' text-anchor='end'>{t}</text>" for t in range(0, 101, 10))
    g += "".join(f"<text x='{X(k)}' y='{y1 + 20}' font-size='12' text-anchor='middle'>{m}</text>" for k, m in enumerate(MONTHS))
    g += f"<polyline fill='none' stroke='#0969da' stroke-width='3' points='{' '.join(f'{X(k)},{Y(v)}' for k, v in enumerate(vals))}'/>"
    g += "".join(f"<circle cx='{X(k)}' cy='{Y(v)}' r='4' fill='#0969da'/>" for k, v in enumerate(vals))
    html = svg_page("Support tickets per month", f"<svg width='{W}' height='{H}' font-family='Helvetica'>{g}</svg>")
    best = MONTHS[vals.index(top)]
    cands = [best] + rng.sample([m for m in MONTHS if m != best], 3); rng.shuffle(cands)
    cap = "A line chart of support tickets per month: " + ", ".join(f"{m} {v}" for m, v in zip(MONTHS, vals)) + "."
    return html, "A line chart.", cap, {
        "peak": choice("In which month were there the most tickets?", cands, best),
        "june": noul("Were there more than 50 tickets in June?", vals[5] > 50),
        "march": score("How many tickets were there in March?", ["0 to 19", "20 to 39", "40 to 59", "60 to 79", "80 to 100"], vals[2] // 20),
    }


COUNTRIES = ["Norway", "Chile", "Kenya", "Japan", "Peru", "Italy", "Egypt", "Canada", "Vietnam", "Poland"]


def bar_chart(rng, i):
    while True:
        vals = [rng.randint(15, 95) for _ in COUNTRIES]
        a, b = rng.sample(range(10), 2)
        if len(set(vals)) == 10 and 3 <= abs(vals[a] - vals[b]) <= 7 and all(abs(v - 60) >= 2 for v in vals): break
    W, H, x0, y0, x1, y1 = 960, 560, 60, 20, 940, 500
    bw = (x1 - x0) / 10
    Y = lambda v: y1 - (y1 - y0) * v / 100
    g = "".join(f"<line x1='{x0}' x2='{x1}' y1='{Y(t)}' y2='{Y(t)}' stroke='#e1e4e8'/><text x='{x0 - 8}' y='{Y(t) + 4}' font-size='12' text-anchor='end'>{t}</text>" for t in range(0, 101, 20))
    for k, (c, v) in enumerate(zip(COUNTRIES, vals)):
        g += f"<rect x='{x0 + k * bw + 10}' y='{Y(v)}' width='{bw - 20}' height='{y1 - Y(v)}' fill='#8250df'/><text x='{x0 + k * bw + bw / 2}' y='{y1 + 18}' font-size='12' text-anchor='middle'>{c}</text>"
    html = svg_page("Renewable share of electricity (%)", f"<svg width='{W}' height='{H}' font-family='Helvetica'>{g}</svg>")
    above = sum(v > 60 for v in vals)
    order = sorted(range(10), key=lambda k: -vals[k])
    second = COUNTRIES[order[1]]
    cands = [second] + [COUNTRIES[k] for k in rng.sample(order[2:], 3)]; rng.shuffle(cands)
    cap = "A bar chart of renewable share of electricity in percent: " + ", ".join(f"{c} {v}" for c, v in zip(COUNTRIES, vals)) + "."
    return html, "A bar chart.", cap, {
        "compare": noul(f"Is {COUNTRIES[a]}'s share higher than {COUNTRIES[b]}'s?", vals[a] > vals[b]),
        "above60": score("How many countries are above 60%?", [str(j) for j in range(11)], above),
        "second": choice("Which country has the second-highest share?", cands, second),
    }


DOT = {"red": (214, 40, 40), "blue": (30, 90, 200), "green": (40, 150, 60)}


def dots(rng, i):
    n = [8, 12, 15, 19, 23, 27, 30, 17][i]
    red = [3, 5, 4, 7, 6, 9, 8, 2][i]
    cols = ["red"] * red + [rng.choice(["blue", "green"]) for _ in range(n - red)]; rng.shuffle(cols)
    W = H = 640
    img = Image.new("RGB", (W, H), (250, 250, 247)); d = ImageDraw.Draw(img)
    placed = []
    for c in cols:
        for _ in range(5000):
            r = rng.randint(14, 22); x, y = rng.randint(r + 6, W - r - 6), rng.randint(r + 6, H - r - 6)
            if all((x - a) ** 2 + (y - b) ** 2 > (r + q + 10) ** 2 for a, b, q in placed): break
        placed.append((x, y, r)); d.ellipse([x - r, y - r, x + r, y + r], fill=DOT[c])
    levels = ["fewer than 10", "10 to 14", "15 to 19", "20 to 24", "25 to 30"]
    counts = {c: cols.count(c) for c in DOT}
    return img, "A picture of colored dots.", f"A picture of {n} dots: {counts['red']} red, {counts['blue']} blue and {counts['green']} green.", {
        "total": score("How many dots are there in total?", levels, 0 if n < 10 else min(4, (n - 10) // 5 + 1)),
        "red": score("How many red dots are there?", [str(j) for j in range(1, 11)], red - 1),
        "more": noul("Are there more blue dots than green dots?", counts["blue"] > counts["green"]),
    }


CITIES = ["Lisbon", "Oslo", "Vienna", "Dublin", "Prague", "Madrid", "Athens", "Zurich", "Warsaw", "Helsinki", "Rome", "Paris"]


def departures(rng, i):
    rows, flights = [], []
    for k, c in enumerate(rng.sample(CITIES, 12)):
        f = f"{rng.choice(['LH', 'BA', 'AF', 'KL', 'OS'])}{rng.randint(100, 999)}"
        gate = f"{rng.choice('ABC')}{rng.randint(1, 40)}"
        st = rng.choice(["On time", "On time", "Boarding", "Delayed 40 min", "Cancelled"])
        flights.append((f, c, gate, st))
        rows.append(f"<tr><td>{8 + k // 3}:{(k * 13) % 60:02d}</td><td>{f}</td><td>{c}</td><td>{gate}</td><td style='color:{'#ffb000' if 'Delayed' in st else '#ff5555' if st == 'Cancelled' else '#9be9a8'}'>{st}</td></tr>")
    body = ("<style>td, th { font-size: 12px; padding: 6px 10px; }</style><table style='background:#0d1117;color:#e6edf3;font-family:Menlo,monospace'><tr><th style='background:#161b22;color:#8b949e'>Time</th>"
            "<th style='background:#161b22;color:#8b949e'>Flight</th><th style='background:#161b22;color:#8b949e'>Destination</th>"
            "<th style='background:#161b22;color:#8b949e'>Gate</th><th style='background:#161b22;color:#8b949e'>Status</th></tr>" + "".join(rows) + "</table>")
    html = page("Departures", body)
    k = rng.randrange(12); f, c, gate, st = flights[k]
    delayed = rng.randrange(12)
    gates = [gate] + rng.sample([g for _, _, g, _ in flights if g != gate] or ["A1", "B2", "C3"], 3); gates = list(dict.fromkeys(gates))
    while len(gates) < 4: gates.append(f"D{len(gates)}")
    rng.shuffle(gates)
    cap = "An airport departures board: " + "; ".join(f"{f} to {c}, gate {g}, {s}" for f, c, g, s in flights) + "."
    return html, "A photo of an airport departures board.", cap, {
        "gate": choice(f"From which gate does the flight to {c} leave?", gates, gate),
        "delayed": noul(f"Is the flight to {flights[delayed][1]} delayed or cancelled?", flights[delayed][3] in ("Delayed 40 min", "Cancelled")),
    }


def label(rng, i):
    kg = [0.4, 2.3, 7.8, 16.5][i]
    express = i % 2 == 1
    to = rng.choice(["Ana Costa, 14 Rua Augusta, 1100-053 Lisboa, Portugal", "Jonas Weber, Hauptstr. 8, 10115 Berlin, Germany",
                     "Chen Wei, 22 Queen St, Toronto ON M5H 2N2, Canada"])
    body = (f"<div class='card' style='width:560px;margin:30px auto;font-size:13px;line-height:1.5'>"
            f"<div style='font-size:22px;font-weight:700'>{'EXPRESS' if express else 'STANDARD'} PARCEL</div>"
            f"<div class='small muted'>Ship to</div><div>{to}</div>"
            f"<div class='small muted' style='margin-top:10px'>Weight</div><div>{kg} kg</div>"
            f"<div class='small muted' style='margin-top:10px'>Tracking</div><div style='font-family:Menlo,monospace'>RX{rng.randint(10**8, 10**9)}DE</div>"
            f"<div style='height:70px;margin-top:12px;background:repeating-linear-gradient(90deg,#000 0 2px,#fff 2px 5px)'></div>"
            f"<div class='small muted' style='margin-top:8px'>Handle with care. Signature {'required' if express else 'not required'} on delivery.</div></div>")
    html = page("Shipping label preview", body, 1024, 700)
    country = to.split(", ")[-1]
    cap = f"A {'express' if express else 'standard'} parcel label to {to}; weight {kg} kg; signature {'required' if express else 'not required'} on delivery."
    return html, "A picture of a shipping label.", cap, {
        "weight": score("How heavy is the parcel?", ["under 1 kg", "1 to 5 kg", "5 to 10 kg", "over 10 kg"], [0, 1, 2, 3][i]),
        "country": choice("To which country is the parcel going?", ["Portugal", "Germany", "Canada", "France"], country),
        "signature": noul("Is a signature required on delivery?", express),
    }


def terms(rng, i):
    renew = i % 2 == 0
    days = [7, 14, 30, 60][i]
    fine = (f"By subscribing you agree to our Terms of Service. Your plan {'renews automatically each month until you cancel' if renew else 'ends after 12 months and does not renew'}. "
            f"You may cancel within {days} days of purchase for a full refund. Prices include VAT where applicable. We may change these terms with 30 days notice.")
    body = (f"<div class='card' style='width:720px;margin:30px auto'><div style='font-size:24px;font-weight:700'>Pro plan</div>"
            f"<div style='font-size:36px;margin:10px 0'>$12<span class='muted' style='font-size:16px'> / month</span></div>"
            f"<div style='background:#0969da;color:#fff;padding:12px;border-radius:6px;text-align:center;font-size:16px;margin:16px 0'>Subscribe</div>"
            f"<div style='font-size:11px;color:#6e7781;line-height:1.4'>{fine}</div></div>")
    html = page("Checkout", body)
    return html, "A screenshot of a checkout page.", f"A checkout page for a $12/month Pro plan with fine print: '{fine}'", {
        "renew": noul("Does the plan renew automatically?", renew),
        "refund": score("How long is the refund window?", ["7 days", "14 days", "30 days", "60 days"], [7, 14, 30, 60].index(days)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260924)
    a = ap.parse_args()
    for d in ("pages", "images"): os.makedirs(f"{a.out}/{d}", exist_ok=True)
    rng = random.Random(a.seed)
    items = []
    for family, fn, n in [("inbox", inbox, 4), ("shop", shop, 4), ("dashboard", dashboard, 4), ("orders", orders, 4), ("line", line_chart, 4),
                          ("bars", bar_chart, 4), ("dots", dots, 8), ("departures", departures, 4), ("label", label, 4), ("terms", terms, 4)]:
        for i in range(n):
            art, context, caption, qs = fn(rng, i)
            name = f"{family}-{i}"
            if isinstance(art, dict):
                open(f"{a.out}/pages/{name}.html", "w").write(art["html"])
                src, size = f"pages/{name}.html", art["size"]
            else:
                art.save(f"{a.out}/images/{name}.png", optimize=True); src, size = None, list(art.size)
            items.append({"id": name, "family": family, "image": f"images/{name}.png", "page": src, "size": size,
                          "license": "Apache-2.0 (generated by kev_web_export.vision_set_v2)", "context": context, "caption": caption, "questions": qs})
    json.dump({"version": "vision-v2", "seed": a.seed, "items": items}, open(f"{a.out}/questions.json", "w"), indent=1, ensure_ascii=False)
    qs = [q for it in items for q in it["questions"].values()]
    by = {t: sum(q["type"] == t for q in qs) for t in ("noul", "choice", "score")}
    print(f"{len(items)} images ({sum(bool(it['page']) for it in items)} pages to render), {len(qs)} questions {by}; "
          f"noul yes {sum(q['label'] for q in qs if q['type'] == 'noul')}/{by['noul']} -> {a.out}")


if __name__ == "__main__":
    main()
