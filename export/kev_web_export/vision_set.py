"""Build eval/vision-v1: image decision questions with known answers, for Kev behind Qwen3.5's vision tower.

    uv run --group vision python -m kev_web_export.vision_set --photos /path/to/skimage-data --out ../eval/vision-v1

Most images are drawn here from a seeded RNG (shapes, bar charts, app screens, receipts, signs, traffic lights), so
their answers are known by construction and their captions are generated from the same parameters. Seven photos
come from scikit-image's sample data (public domain or CC0, see LICENSES.md), with hand-written questions and captions.
Text is drawn with Pillow's bundled default font (`ImageFont.load_default(size)`), so the images depend only on the
Pillow version; the PNGs are committed and this script documents how they were made.

Every item carries a `context` (neutral text that every condition shows, it never gives the answer away) and a
`caption` (a text description with everything the questions need: the text-only upper bound). Labels index the
question's options in kev.api order: noul 0 = no, 1 = yes; choice = criteria order; score = level index."""
import argparse, json, os, random, shutil
from PIL import Image, ImageDraw, ImageFont

W = H = 448
COLORS = {"red": (220, 40, 40), "blue": (40, 90, 220), "green": (40, 170, 70), "yellow": (240, 200, 30)}


def font(size):
    return ImageFont.load_default(size=size)


def noul(instr, yes):
    return {"type": "noul", "instructions": instr, "label": int(yes)}


def choice(instr, options, answer):
    return {"type": "choice", "instructions": instr, "criteria": {o: None for o in options}, "label": options.index(answer)}


def score(instr, levels, index):
    return {"type": "score", "instructions": instr, "criteria": levels, "label": index}


def shapes(rng, i):
    n = [2, 3, 4, 5, 6, 7, 3, 5][i]
    major = list(COLORS)[i % 4]
    cols = [major] * (n // 2 + 1) + [rng.choice([c for c in COLORS if c != major]) for _ in range(n - n // 2 - 1)]
    kinds = [rng.choice(["circle", "square"]) for _ in range(n)]
    if i % 2 == 0: kinds[rng.randrange(n)] = "triangle"
    else: kinds = [k if k != "triangle" else "circle" for k in kinds]
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    placed = []
    for col, kind in zip(cols, kinds):
        for _ in range(1000):
            r = rng.randint(28, 42); x, y = rng.randint(r + 8, W - r - 8), rng.randint(r + 8, H - r - 8)
            if all((x - a) ** 2 + (y - b) ** 2 > (r + c + 14) ** 2 for a, b, c in placed): break
        placed.append((x, y, r))
        box = [x - r, y - r, x + r, y + r]
        if kind == "circle": d.ellipse(box, fill=COLORS[col])
        elif kind == "square": d.rectangle(box, fill=COLORS[col])
        else: d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=COLORS[col])
    desc = ", ".join(f"a {c} {k}" for c, k in zip(cols, kinds))
    has_tri = "triangle" in kinds
    return img, "A picture of simple shapes.", f"A white picture with {n} shapes: {desc}.", {
        "count": score("How many shapes are in the picture?", [str(k) for k in range(1, 9)], n - 1),
        "color": choice("What color are most of the shapes?", list(COLORS), major),
        "triangle": noul("Is there a triangle in the picture?", has_tri),
    }


FRUITS = ["apples", "bananas", "cherries", "dates"]


def bars(rng, i):
    while True:
        vals = [rng.choice(range(5, 96, 5)) for _ in FRUITS]
        top = max(vals)
        if vals.count(top) == 1 and all(abs(top - b) >= 8 for b in (25, 50, 75)) and len(set(vals)) == 4: break
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 60, 50, W - 20, H - 60
    d.text((W // 2, 22), "Fruit sold this week", fill="black", font=font(20), anchor="mm")
    for t in range(0, 101, 25):
        y = y1 - (y1 - y0) * t / 100
        d.line([(x0, y), (x1, y)], fill=(210, 210, 210)); d.text((x0 - 8, y), str(t), fill="black", font=font(14), anchor="rm")
    bw = (x1 - x0) / len(FRUITS)
    for k, (f, v) in enumerate(zip(FRUITS, vals)):
        bx = x0 + k * bw + bw * 0.2
        d.rectangle([bx, y1 - (y1 - y0) * v / 100, bx + bw * 0.6, y1], fill=(70, 110, 190))
        d.text((bx + bw * 0.3, y1 + 16), f, fill="black", font=font(15), anchor="mm")
    d.line([(x0, y0), (x0, y1), (x1, y1)], fill="black", width=2)
    a, b = rng.sample(range(4), 2)
    best = FRUITS[vals.index(top)]
    return img, "A bar chart of fruit sales.", "A bar chart titled 'Fruit sold this week': " + ", ".join(f"{f} {v}" for f, v in zip(FRUITS, vals)) + ".", {
        "best": choice("Which fruit sold the most?", FRUITS, best),
        "compare": noul(f"Did {FRUITS[a]} sell more than {FRUITS[b]}?", vals[a] > vals[b]),
        "top": score("How many units did the best-selling fruit sell?", ["under 25", "25 to 50", "50 to 75", "over 75"], min(3, top // 25)),
    }


def window(title):
    img = Image.new("RGB", (W, H), (245, 246, 248)); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 44], fill=(52, 58, 70)); d.text((16, 22), title, fill="white", font=font(18), anchor="lm")
    return img, d


def button(d, box, label, primary=True):
    d.rounded_rectangle(box, 8, fill=(40, 100, 220) if primary else (225, 228, 232))
    d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, fill="white" if primary else "black", font=font(16), anchor="mm")


SCREENS = ["login", "payment result", "settings", "task list"]


def screen(rng, i):
    kind = ["payment", "payment", "payment", "payment", "settings", "settings", "tasks", "tasks", "login", "login"][i]
    if kind == "payment":
        ok = i % 2 == 0
        amount = rng.choice([12.5, 48.0, 89.99, 230.0])
        img, d = window("Checkout")
        d.ellipse([W / 2 - 40, 90, W / 2 + 40, 170], fill=(40, 170, 70) if ok else (210, 50, 50))
        d.text((W / 2, 130), "OK" if ok else "!", fill="white", font=font(34), anchor="mm")
        d.text((W / 2, 210), "Payment successful" if ok else "Payment failed", fill="black", font=font(26), anchor="mm")
        d.text((W / 2, 250), f"${amount:.2f} charged to Visa 4242" if ok else "Your card was declined.", fill=(80, 80, 80), font=font(17), anchor="mm")
        button(d, [W / 2 - 110, 320, W / 2 + 110, 370], "Continue shopping" if ok else "Try again")
        cap = (f"A checkout screen with a green check mark: 'Payment successful', ${amount:.2f} charged to Visa 4242, and a 'Continue shopping' button." if ok
               else "A checkout screen with a red exclamation mark: 'Payment failed. Your card was declined.' and a 'Try again' button.")
        return img, "A screenshot from an online shop.", cap, {
            "paid": noul("Did the payment go through?", ok),
            "screen": choice("What kind of screen is this?", SCREENS, "payment result"),
            "error": noul("Does the screen show an error?", not ok),
        }
    if kind == "settings":
        dark, notif = (True, False) if i == 4 else (False, True)
        img, d = window("Settings")
        for k, (name, on) in enumerate([("Dark mode", dark), ("Notifications", notif), ("Auto-update", True)]):
            y = 100 + 70 * k
            d.text((40, y), name, fill="black", font=font(20), anchor="lm")
            d.rounded_rectangle([W - 110, y - 16, W - 50, y + 16], 16, fill=(40, 170, 70) if on else (190, 190, 190))
            cx = W - 66 if on else W - 94
            d.ellipse([cx - 13, y - 13, cx + 13, y + 13], fill="white")
            d.text((W - 124, y), "On" if on else "Off", fill=(90, 90, 90), font=font(15), anchor="rm")
        state = lambda on: "on" if on else "off"
        return img, "A screenshot of an app.", f"A settings screen with switches: Dark mode {state(dark)}, Notifications {state(notif)}, Auto-update on.", {
            "dark": noul("Is dark mode enabled?", dark),
            "notif": noul("Are notifications turned on?", notif),
            "screen": choice("What kind of screen is this?", SCREENS, "settings"),
        }
    if kind == "tasks":
        tasks = ["Buy milk", "Call the bank", "Book dentist", "Pay rent", "Water plants"]
        done = [1, 4][i - 6]
        marks = [True] * done + [False] * (5 - done); rng.shuffle(marks)
        img, d = window("My tasks")
        for k, (t, m) in enumerate(zip(tasks, marks)):
            y = 90 + 60 * k
            d.rectangle([40, y - 14, 68, y + 14], outline="black", width=2, fill=(40, 170, 70) if m else "white")
            if m: d.line([(45, y), (52, y + 8), (64, y - 9)], fill="white", width=4)
            d.text((88, y), t, fill=(140, 140, 140) if m else "black", font=font(20), anchor="lm")
        return img, "A screenshot of an app.", "A to-do list app with five tasks: " + ", ".join(f"{t} ({'done' if m else 'open'})" for t, m in zip(tasks, marks)) + ".", {
            "done": score("How many tasks are marked done?", [str(k) for k in range(6)], done),
            "rent": noul("Is 'Pay rent' marked done?", marks[3]),
            "screen": choice("What kind of screen is this?", SCREENS, "task list"),
        }
    err = i == 9
    img, d = window("Sign in")
    for k, (label, value) in enumerate([("Email", "sam@example.com"), ("Password", "********")]):
        y = 100 + 90 * k
        d.text((60, y), label, fill="black", font=font(17), anchor="lm")
        d.rectangle([60, y + 16, W - 60, y + 56], outline=(210, 50, 50) if err and k == 1 else (160, 160, 160), width=2, fill="white")
        d.text((72, y + 36), value, fill="black", font=font(17), anchor="lm")
    if err: d.text((60, 300), "Incorrect password. 2 attempts left.", fill=(210, 50, 50), font=font(17), anchor="lm")
    button(d, [60, 340, W - 60, 390], "Sign in")
    cap = "A sign-in form with email sam@example.com and a filled password field" + (", and a red message: 'Incorrect password. 2 attempts left.'" if err else ", and no messages.")
    return img, "A screenshot of an app.", cap, {
        "error": noul("Does the screen show an error?", err),
        "screen": choice("What kind of screen is this?", SCREENS, "login"),
    }


def receipt(rng, i):
    total = [8.4, 36.75, 64.2, 142.9][i]
    cash = i % 2 == 1
    img = Image.new("RGB", (W, H), (230, 230, 225)); d = ImageDraw.Draw(img)
    d.rectangle([90, 20, W - 90, H - 20], fill="white")
    d.text((W / 2, 50), "CORNER MARKET", fill="black", font=font(22), anchor="mm")
    items = rng.sample(["Bread", "Milk", "Eggs", "Coffee", "Cheese", "Apples", "Rice", "Soap"], 3)
    parts = [round(total * f, 2) for f in (0.5, 0.3)]; parts.append(round(total - sum(parts), 2))
    for k, (it, p) in enumerate(zip(items, parts)):
        d.text((110, 100 + 34 * k), it, fill="black", font=font(18), anchor="lm"); d.text((W - 110, 100 + 34 * k), f"{p:.2f}", fill="black", font=font(18), anchor="rm")
    d.line([(110, 215), (W - 110, 215)], fill="black", width=2)
    d.text((110, 245), "TOTAL", fill="black", font=font(22), anchor="lm"); d.text((W - 110, 245), f"${total:.2f}", fill="black", font=font(22), anchor="rm")
    d.text((110, 290), "Paid: " + ("CASH" if cash else "VISA ****4242"), fill="black", font=font(18), anchor="lm")
    d.text((W / 2, 360), "Thank you!", fill="black", font=font(18), anchor="mm")
    return img, "A photo of a shopping receipt.", (f"A receipt from Corner Market: {items[0]} {parts[0]:.2f}, {items[1]} {parts[1]:.2f}, {items[2]} {parts[2]:.2f}, "
                                                   f"TOTAL ${total:.2f}, paid {'in cash' if cash else 'by Visa card'}."), {
        "total": score("How much was spent in total?", ["under $20", "$20 to $50", "$50 to $100", "over $100"], [0, 1, 2, 3][i]),
        "cash": noul("Was this paid in cash?", cash),
    }


def sign(rng, i):
    is_open = i % 2 == 0
    img = Image.new("RGB", (W, H), (120, 90, 60)); d = ImageDraw.Draw(img)
    d.rounded_rectangle([60, 120, W - 60, 300], 16, fill=(30, 130, 60) if is_open else (180, 30, 30))
    d.text((W / 2, 185), "OPEN" if is_open else "CLOSED", fill="white", font=font(64), anchor="mm")
    d.text((W / 2, 260), "Come in!" if is_open else "Back at 9:00", fill="white", font=font(24), anchor="mm")
    return img, "A photo of a sign on a shop door.", f"A {'green' if is_open else 'red'} sign on a shop door reading '{'OPEN' if is_open else 'CLOSED'}' and '{'Come in!' if is_open else 'Back at 9:00'}'.", {
        "open": noul("Is the shop open right now?", is_open),
    }


def traffic(rng, i):
    lit = ["red", "yellow", "green", "red"][i]
    img = Image.new("RGB", (W, H), (150, 190, 230)); d = ImageDraw.Draw(img)
    d.rectangle([W / 2 - 12, 330, W / 2 + 12, H], fill=(60, 60, 60))
    d.rounded_rectangle([W / 2 - 60, 40, W / 2 + 60, 340], 20, fill=(35, 35, 35))
    on = {"red": (240, 30, 30), "yellow": (250, 200, 20), "green": (30, 220, 80)}
    for k, c in enumerate(["red", "yellow", "green"]):
        y = 90 + 100 * k
        d.ellipse([W / 2 - 38, y - 38, W / 2 + 38, y + 38], fill=on[c] if c == lit else (70, 70, 70))
    return img, "A photo of a traffic light.", f"A traffic light with the {lit} light on and the other two off.", {
        "light": choice("Which light is on?", ["red", "yellow", "green"], lit),
        "stop": noul("Should a car approaching this light stop?", lit == "red"),
    }


PHOTOS = [   # (file, license, context, caption, questions)
    ("astronaut.png", "Public domain (NASA); scikit-image sample data",
     "A photo.", "A studio portrait of a smiling woman astronaut in an orange NASA flight suit, a US flag behind her, a black helmet on a table in front of her and a space shuttle model on the right.",
     {"person": noul("Is there a person in the photo?", True), "animal": noul("Is there an animal in the photo?", False),
      "job": choice("What is the person's job?", ["astronaut", "chef", "doctor", "firefighter"], "astronaut"),
      "suit": choice("What color is the suit?", ["orange", "white", "blue", "green"], "orange")}),
    ("chelsea.png", "CC0 (Stefan van der Walt); scikit-image sample data",
     "A photo.", "A close-up of a tabby cat's face with green-yellow eyes and a pink nose.",
     {"animal": choice("What animal is this?", ["cat", "dog", "horse", "bird"], "cat"), "person": noul("Is there a person in the photo?", False)}),
    ("coffee.png", "CC0 (Rachel Michetti); scikit-image sample data",
     "A photo.", "A red cup of espresso with crema on a red saucer with a metal spoon, on a wooden table.",
     {"drink": choice("What drink is shown?", ["coffee", "orange juice", "water", "wine"], "coffee"),
      "spoon": noul("Is there a spoon?", True), "empty": noul("Is the cup empty?", False)}),
    ("rocket.jpg", "Public domain (SpaceX); scikit-image sample data",
     "A photo.", "A white rocket standing on its launch pad between lightning towers at dusk, lit by floodlights, under a dark blue sky.",
     {"vehicle": choice("What vehicle is shown?", ["rocket", "airplane", "ship", "car"], "rocket"), "night": noul("Was this photo taken at night or dusk?", True)}),
    ("coins.png", "No known copyright restrictions (Brooklyn Museum); scikit-image sample data",
     "A photo.", "A black and white photo of 24 ancient Greek coins laid out in four rows of six on a dark background.",
     {"count": score("How many coins are there?", ["fewer than 10", "10 to 20", "21 to 30", "more than 30"], 2),
      "coins": noul("Is this a photo of coins?", True), "color": noul("Is the photo in color?", False)}),
    ("hubble_deep_field.jpg", "Public domain (NASA); scikit-image sample data",
     "A photo.", "A telescope image of deep space: many small galaxies of different colors on a black background.",
     {"scene": choice("Where was this picture taken?", ["outer space", "the ocean", "a forest", "a city at night"], "outer space"),
      "people": noul("Are there people in the picture?", False)}),
    ("horse.png", "CC0 (Andreas Preuss); scikit-image sample data",
     "A picture.", "A black silhouette of a standing horse on a white background.",
     {"animal": choice("What animal is shown?", ["horse", "dog", "cat", "cow"], "horse"), "color": noul("Is the picture in color?", False)}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True, help="directory with the scikit-image sample files listed in PHOTOS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260923)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/images", exist_ok=True)
    rng = random.Random(a.seed)
    items = []
    for family, fn, n in [("shapes", shapes, 8), ("chart", bars, 6), ("screen", screen, 10), ("receipt", receipt, 4), ("sign", sign, 2), ("traffic", traffic, 4)]:
        for i in range(n):
            img, context, caption, qs = fn(rng, i)
            name = f"{family}-{i}.png"
            img.save(f"{a.out}/images/{name}", optimize=True)
            items.append({"id": f"{family}-{i}", "family": family, "image": f"images/{name}", "license": "Apache-2.0 (generated by kev_web_export.vision_set)",
                          "context": context, "caption": caption, "questions": qs})
    for f, lic, context, caption, qs in PHOTOS:
        shutil.copy(f"{a.photos}/{f}", f"{a.out}/images/{f}")
        items.append({"id": os.path.splitext(f)[0], "family": "photo", "image": f"images/{f}", "license": lic, "context": context, "caption": caption, "questions": qs})
    json.dump({"version": "vision-v1", "seed": a.seed, "items": items}, open(f"{a.out}/questions.json", "w"), indent=1, ensure_ascii=False)
    qs = [q for it in items for q in it["questions"].values()]
    by = {t: sum(q["type"] == t for q in qs) for t in ("noul", "choice", "score")}
    print(f"{len(items)} images, {len(qs)} questions {by}; noul yes {sum(q['label'] for q in qs if q['type'] == 'noul')}/{by['noul']} -> {a.out}")


if __name__ == "__main__":
    main()
