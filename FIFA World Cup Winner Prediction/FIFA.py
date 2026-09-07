
# ================================================
# Project: FIFA 2026 World Cup Winner Predictor
# Type   : Classification (Logistic Regression)
# Dataset: results.csv (47,000+ matches)
# ================================================

# ================================================
# Project: FIFA 2026 World Cup Winner Predictor
# Type   : Classification (Random Forest)
# Dataset: results.csv
# ================================================
# ================================================
# Project : FIFA 2026 World Cup Winner Predictor
# Model   : Binary Classification (GradientBoosting + ELO)
# Dataset : results.csv
# ================================================

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── 1. Load ───────────────────────────────────────
DATA_FILE = Path(__file__).resolve().parent / "results.csv"
df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"✅ Dataset: {len(df):,} matches loaded")

# ── 2. Team Stats (Competitive 2015+) ─────────────
comp   = df[df['tournament'] != 'Friendly'].copy()
recent = comp[comp['date'] >= '2015-01-01'].copy()

def calc_stats(team, data):
    h = data[data['home_team'] == team]
    a = data[data['away_team'] == team]
    n = len(h) + len(a)
    if n < 5: return None
    hw = len(h[h['home_score'] > h['away_score']])
    aw = len(a[a['away_score'] > a['home_score']])
    hd = len(h[h['home_score'] == h['away_score']])
    ad = len(a[a['away_score'] == a['home_score']])
    gf = h['home_score'].sum() + a['away_score'].sum()
    ga = h['away_score'].sum() + a['home_score'].sum()
    dt = data['date'].max()
    r  = data[data['date'] >= dt - pd.DateOffset(years=2)]
    rh = r[r['home_team'] == team]; ra = r[r['away_team'] == team]
    rt = max(len(rh)+len(ra), 1)
    rw = len(rh[rh['home_score'] > rh['away_score']]) + \
         len(ra[ra['away_score'] > ra['home_score']])
    rf = rh['home_score'].sum() + ra['away_score'].sum()
    rg = rh['away_score'].sum() + ra['home_score'].sum()
    return {
        'win_rate'        : (hw+aw)/n,
        'draw_rate'       : (hd+ad)/n,
        'avg_gf'          : gf/n,
        'avg_ga'          : ga/n,
        'goal_diff'       : (gf-ga)/n,
        'recent_win_rate' : rw/rt,
        'recent_goal_diff': (rf-rg)/rt,
        'recent_avg_gf'   : rf/rt,
    }

team_stats = {}
for t in pd.concat([recent['home_team'], recent['away_team']]).unique():
    s = calc_stats(t, recent)
    if s: team_stats[t] = s
print(f"✅ Stats for {len(team_stats)} teams computed")

# ── 3. ELO + Training Data (One Pass) ─────────────
# ELO is computed from ALL 49k matches chronologically
# Training uses only 2010+ decisive matches (no draws)
DEFL = {'win_rate':0.35,'draw_rate':0.25,'avg_gf':1.1,'avg_ga':1.4,
        'goal_diff':-0.3,'recent_win_rate':0.35,'recent_goal_diff':-0.3,'recent_avg_gf':1.0}
K    = 32
elo  = defaultdict(lambda: 1500)
X_list, y_list = [], []

for _, row in df.iterrows():
    h, a   = row['home_team'], row['away_team']
    he, ae = elo[h], elo[a]
    s1, s2 = team_stats.get(h, DEFL), team_stats.get(a, DEFL)

    # Only 2010+ decisive matches for training
    if row['date'].year >= 2010 and row['home_score'] != row['away_score']:
        X_list.append([
            he, ae, he-ae,
            s1['win_rate'], s1['avg_gf'],  s1['avg_ga'],   s1['goal_diff'],
            s1['recent_win_rate'], s1['recent_goal_diff'],
            s1['recent_avg_gf'],   s1['draw_rate'],
            s2['win_rate'], s2['avg_gf'],  s2['avg_ga'],   s2['goal_diff'],
            s2['recent_win_rate'], s2['recent_goal_diff'],
            s2['recent_avg_gf'],   s2['draw_rate'],
            s1['win_rate']         - s2['win_rate'],
            s1['goal_diff']        - s2['goal_diff'],
            s1['recent_win_rate']  - s2['recent_win_rate'],
            s1['recent_goal_diff'] - s2['recent_goal_diff'],
            s1['avg_gf']           - s2['avg_gf'],
        ])
        y_list.append(1 if row['home_score'] > row['away_score'] else 0)

    # ELO update after recording pre-match ELO
    exp_h = 1 / (1 + 10**((ae-he)/400))
    sh = 1.0 if row['home_score']>row['away_score'] else \
         0.0 if row['home_score']<row['away_score'] else 0.5
    elo[h] += K*(sh - exp_h)
    elo[a] += K*((1-sh) - (1-exp_h))

final_elo = dict(elo)
X, y = np.array(X_list), np.array(y_list)
print(f"✅ Training samples : {len(X):,} (decisive matches only)")
print(f"✅ ELO computed for : {len(final_elo)} teams")

# ── 4. Train GradientBoosting ─────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)

print("\n⏳ Training model... (30-60 seconds please wait)")
model = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=4, subsample=0.8,
    min_samples_leaf=10, random_state=42)
model.fit(Xtr, y_train)
yp = model.predict(Xte)

print("\n════════ Model Metrics ════════")
print(f"Accuracy  : {accuracy_score(y_test, yp)*100:.1f}%")
print(f"Precision : {precision_score(y_test, yp, average='weighted', zero_division=0)*100:.1f}%")
print(f"Recall    : {recall_score(y_test, yp, average='weighted', zero_division=0)*100:.1f}%")
print(f"F1 Score  : {f1_score(y_test, yp, average='weighted', zero_division=0)*100:.1f}%")
print("(Note: R² = Regression only. Classification → Accuracy, F1)")

# ── 5. Predict Function ───────────────────────────
ci = {c:i for i,c in enumerate(model.classes_)}

def draw_prob(elo_d):
    # Closer ELO → more likely draw
    return max(0.08, 0.27 * np.exp(-abs(elo_d)/450))

def predict_match(t1, t2):
    e1 = final_elo.get(t1, 1500)
    e2 = final_elo.get(t2, 1500)
    s1 = team_stats.get(t1, DEFL)
    s2 = team_stats.get(t2, DEFL)
    feat = scaler.transform([[
        e1, e2, e1-e2,
        s1['win_rate'], s1['avg_gf'],  s1['avg_ga'],   s1['goal_diff'],
        s1['recent_win_rate'], s1['recent_goal_diff'],
        s1['recent_avg_gf'],   s1['draw_rate'],
        s2['win_rate'], s2['avg_gf'],  s2['avg_ga'],   s2['goal_diff'],
        s2['recent_win_rate'], s2['recent_goal_diff'],
        s2['recent_avg_gf'],   s2['draw_rate'],
        s1['win_rate']         - s2['win_rate'],
        s1['goal_diff']        - s2['goal_diff'],
        s1['recent_win_rate']  - s2['recent_win_rate'],
        s1['recent_goal_diff'] - s2['recent_goal_diff'],
        s1['avg_gf']           - s2['avg_gf'],
    ]])
    prob = model.predict_proba(feat)[0]
    p1d  = prob[ci.get(1, 1)]   # t1 wins (decisive)
    p2d  = prob[ci.get(0, 0)]   # t2 wins (decisive)
    drw  = draw_prob(e1-e2)
    p1   = p1d*(1-drw)
    p2   = p2d*(1-drw)
    tot  = p1+drw+p2
    return p1/tot, drw/tot, p2/tot

# ── 6. Strength Score (ELO-based) ─────────────────
def strength(t):
    e  = final_elo.get(t, 1500)
    s  = team_stats.get(t, DEFL)
    en = (e - 1000) / 600           # normalize ELO
    fm = (s['recent_win_rate']*0.5 +
          s['recent_goal_diff']*0.3 +
          s['win_rate']*0.2)
    return en*0.65 + fm*0.35

# ── 7. 48 Teams ───────────────────────────────────
# ── 6. FIFA 2026 — 48 Teams ──────────────────────
teams_48 = [
    'Japan','South Korea','Australia',
    'Saudi Arabia','Iraq','Jordan','Uzbekistan',
    'Germany','France','England','Portugal',
    'Netherlands','Belgium','Croatia','Switzerland','Austria',
    'Scotland','Turkey','Serbia','Poland','Denmark','Italy',
    'Spain','Brazil','Argentina','Uruguay','Colombia','Ecuador','Paraguay',
    'United States','Mexico','Canada','Costa Rica','Panama','Jamaica',
    'Iran','Morocco','Senegal','Nigeria','Cameroon','Ghana',
    'Egypt','Algeria','Ivory Coast','DR Congo',
    'New Zealand','Venezuela','Bolivia',
]

ranked_48 = sorted(teams_48, key=strength, reverse=True)

print("\n" + "═"*60)
print("         🏆 FIFA 2026 World Cup Prediction")
print("═"*60)
print("\n📊 Top 10 Strongest Teams (ELO + Recent Form):")
for i, t in enumerate(ranked_48[:10], 1):
    e   = final_elo.get(t, 1500)
    sc  = strength(t)
    bar = "█" * int(sc * 18)
    print(f"  {i:2}. {t:<22}  ELO:{e:>6,.0f}  {bar}  ({sc:.3f})")

# Keep the tournament shape valid if the list is edited later.
if len(teams_48) != 48:
    raise ValueError(f"Expected 48 teams, found {len(teams_48)}")

# ── 8. Group Draw (Pot-Based) ─────────────────────
print("\n\n📋 Group Draw — 12 Groups × 4 Teams:")
groups = {}
for i in range(12):
    g = chr(65+i)
    groups[g] = [ranked_48[i], ranked_48[i+12],
                 ranked_48[i+24], ranked_48[i+36]]
    print(f"  Group {g}: {' | '.join(groups[g])}")

# ── 9. Group Stage (Round Robin) ──────────────────
print("\n\n⚽ Group Stage Simulation:")
group_standings, third_data = {}, []

for gname, teams in groups.items():
    tbl = {t:{'pts':0,'gf':0,'ga':0,'w':0,'d':0,'l':0} for t in teams}
    for i in range(4):
        for j in range(i+1, 4):
            t1, t2     = teams[i], teams[j]
            p1, pd_, p2 = predict_match(t1, t2)
            if p1 > p2 and p1 > pd_:
                tbl[t1]['pts']+=3; tbl[t1]['w']+=1; tbl[t2]['l']+=1
                tbl[t1]['gf']+=2;  tbl[t1]['ga']+=1
                tbl[t2]['gf']+=1;  tbl[t2]['ga']+=2
            elif p2 > p1 and p2 > pd_:
                tbl[t2]['pts']+=3; tbl[t2]['w']+=1; tbl[t1]['l']+=1
                tbl[t2]['gf']+=2;  tbl[t2]['ga']+=1
                tbl[t1]['gf']+=1;  tbl[t1]['ga']+=2
            else:
                tbl[t1]['pts']+=1; tbl[t1]['d']+=1
                tbl[t2]['pts']+=1; tbl[t2]['d']+=1
                tbl[t1]['gf']+=1;  tbl[t1]['ga']+=1
                tbl[t2]['gf']+=1;  tbl[t2]['ga']+=1

    st = sorted(tbl.items(),
                key=lambda x:(x[1]['pts'], x[1]['gf']-x[1]['ga'], x[1]['gf']),
                reverse=True)
    group_standings[gname] = st

    print(f"\n  Group {gname}:")
    print(f"  {'Team':<22} W  D  L  GF GA  GD  Pts   Status")
    print(f"  {'─'*60}")
    for rk, (t, s) in enumerate(st, 1):
        gd  = s['gf']-s['ga']
        tag = "✅ Advance" if rk<=2 else "〽️ 3rd" if rk==3 else "❌ Out"
        print(f"  {t:<22} {s['w']}  {s['d']}  {s['l']}"
              f"  {s['gf']:2}  {s['ga']:2}  {gd:+2d}  {s['pts']}    {tag}")

    th = st[2]
    third_data.append((th[0], th[1]['pts'],
                       th[1]['gf']-th[1]['ga'],
                       th[1]['gf'], gname))

# ── 10. Best 8 Third-Place ────────────────────────
third_data.sort(key=lambda x:(x[1],x[2],x[3]), reverse=True)
best8   = [x[0] for x in third_data[:8]]
elim3   = [x[0] for x in third_data[8:]]
winners = [group_standings[g][0][0] for g in groups]
runners = [group_standings[g][1][0] for g in groups]

print(f"\n\n📌 Best 8 Third-Place Teams → Round of 32:")
for x in third_data[:8]:
    print(f"  ✅ {x[0]:<22} Group {x[4]}  Pts:{x[1]}  GD:{x[2]:+d}")
print(f"\n  ❌ Eliminated 3rd: {', '.join(elim3)}")

r32 = sorted(winners + runners + best8, key=strength, reverse=True)
print(f"\n\n🏆 Round of 32 — 32 Teams:")
for i, t in enumerate(r32, 1):
    e = final_elo.get(t, 1500)
    print(f"  {i:2}. {t:<22}  ELO: {e:>6,.0f}")

# ── 11. Knockout Rounds ───────────────────────────
def knockout(teams, rname):
    print(f"\n🔷 {rname}  ({len(teams)} → {len(teams)//2} teams):")
    print(f"  {'Match':<46} Winner          Prob")
    print(f"  {'─'*64}")
    wl = []
    for i in range(0, len(teams)-1, 2):
        t1, t2      = teams[i], teams[i+1]
        p1, pd_, p2  = predict_match(t1, t2)
        w           = t1 if p1 >= p2 else t2
        wp          = max(p1, p2)*100
        m           = f"{t1} vs {t2}"
        print(f"  {m:<46} ✅ {w:<15} ({wp:.0f}%)")
        wl.append(w)
    return wl

cur = r32
for rn in ["Round of 32","Round of 16","Quarter-Finals","Semi-Finals","Final"]:
    cur = knockout(cur, rn)

print("\n" + "═"*60)
print(f"   🥇  PREDICTED WINNER: {cur[0]} 🏆")
print("═"*60)

# ── 12. Custom Match Predictor ────────────────────
print("\n\n🔮 Custom Match Predictor")
print("─"*44)
try:
    t1 = input("Team 1 (e.g. Brazil): ").strip()
    t2 = input("Team 2 (e.g. France): ").strip()
except EOFError:
    t1, t2 = "Brazil", "France"
    print("No interactive input found; using Brazil vs France.")

if not t1 or not t2:
    t1, t2 = "Brazil", "France"
    print("Empty team name; using Brazil vs France.")

p1, pd_, p2 = predict_match(t1, t2)
e1 = final_elo.get(t1, 1500)
e2 = final_elo.get(t2, 1500)
print(f"\n  {t1:<24}  ELO:{e1:>6,.0f}  Win: {p1*100:.1f}%")
print(f"  {'Draw':<24}               Draw:{pd_*100:.1f}%")
print(f"  {t2:<24}  ELO:{e2:>6,.0f}  Win: {p2*100:.1f}%")
print(f"\n  🏆 Predicted Winner: {t1 if p1>p2 else t2}")
print("─"*44)