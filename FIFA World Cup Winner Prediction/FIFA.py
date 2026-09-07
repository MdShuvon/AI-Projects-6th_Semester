
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


# ================================================
# Project : FIFA 2026 World Cup Winner Predictor
# Model   : GradientBoosting + ELO + H2H + Momentum
# Dataset : results.csv
# ================================================

# import pandas as pd
# import numpy as np
# from collections import defaultdict
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (accuracy_score, precision_score,
#                              recall_score, f1_score)
# import warnings
# warnings.filterwarnings('ignore')

# # ════════════════════════════════════════════════
# # 1. LOAD & PREPARE
# # ════════════════════════════════════════════════
# df = pd.read_csv("results.csv")
# df['date'] = pd.to_datetime(df['date'])
# df = df.sort_values('date').reset_index(drop=True)
# print(f"✅ Total matches loaded : {len(df):,}")

# # Tournament weight — important match = more weight
# tournament_weights = {
#     'FIFA World Cup'               : 5.0,
#     'UEFA Euro'                    : 4.0,
#     'Copa América'                 : 4.0,
#     'Africa Cup of Nations'        : 3.5,
#     'AFC Asian Cup'                : 3.5,
#     'CONCACAF Gold Cup'            : 3.0,
#     'UEFA Nations League'          : 3.0,
#     'FIFA World Cup qualification' : 3.0,
#     'Friendly'                     : 1.0,
# }
# def get_weight(t):
#     for key, w in tournament_weights.items():
#         if key.lower() in str(t).lower():
#             return w
#     return 2.0

# df['weight'] = df['tournament'].apply(get_weight)

# # ════════════════════════════════════════════════
# # 2. ELO RATING (from all 49k matches)
# # ════════════════════════════════════════════════
# K_BASE = 30
# elo    = defaultdict(lambda: 1500.0)

# def expected(r1, r2):
#     return 1 / (1 + 10**((r2 - r1) / 400))

# for _, row in df.iterrows():
#     h, a   = row['home_team'], row['away_team']
#     k      = K_BASE * row['weight']
#     exp_h  = expected(elo[h], elo[a])
#     sh     = (1.0 if row['home_score'] > row['away_score'] else
#               0.0 if row['home_score'] < row['away_score'] else 0.5)
#     elo[h] += k * (sh - exp_h)
#     elo[a] += k * ((1 - sh) - (1 - exp_h))

# final_elo = dict(elo)
# print(f"✅ ELO computed for {len(final_elo)} teams")

# # ════════════════════════════════════════════════
# # 3. TEAM STATS (Competitive 2015+)
# # ════════════════════════════════════════════════
# comp   = df[df['weight'] > 1.0].copy()
# recent = comp[comp['date'] >= '2015-01-01'].copy()

# def calc_stats(team, data):
#     h = data[data['home_team'] == team]
#     a = data[data['away_team'] == team]
#     n = len(h) + len(a)
#     if n < 5: return None

#     hw = len(h[h['home_score'] > h['away_score']])
#     aw = len(a[a['away_score'] > a['home_score']])
#     hd = len(h[h['home_score'] == h['away_score']])
#     ad = len(a[a['away_score'] == a['home_score']])
#     gf = h['home_score'].sum() + a['away_score'].sum()
#     ga = h['away_score'].sum() + a['home_score'].sum()

#     # Recent 2-year form
#     dt  = data['date'].max()
#     r   = data[data['date'] >= dt - pd.DateOffset(years=2)]
#     rh  = r[r['home_team'] == team]
#     ra  = r[r['away_team'] == team]
#     rt  = max(len(rh) + len(ra), 1)
#     rw  = (len(rh[rh['home_score'] > rh['away_score']]) +
#            len(ra[ra['away_score'] > ra['home_score']]))
#     rf  = rh['home_score'].sum() + ra['away_score'].sum()
#     rg  = rh['away_score'].sum() + ra['home_score'].sum()

#     # Momentum — last 5 matches
#     all_m = pd.concat([
#         h[['date','home_score','away_score']].rename(
#             columns={'home_score':'gf','away_score':'ga'}),
#         a[['date','away_score','home_score']].rename(
#             columns={'away_score':'gf','home_score':'ga'}),
#     ]).sort_values('date').tail(5)
#     mom = 0.0
#     if len(all_m):
#         pts = sum(3 if r.gf > r.ga else 1 if r.gf == r.ga else 0
#                   for _, r in all_m.iterrows())
#         mom = pts / (len(all_m) * 3)

#     return {
#         'win_rate'        : (hw + aw) / n,
#         'draw_rate'       : (hd + ad) / n,
#         'avg_gf'          : gf / n,
#         'avg_ga'          : ga / n,
#         'goal_diff'       : (gf - ga) / n,
#         'recent_win_rate' : rw / rt,
#         'recent_goal_diff': (rf - rg) / rt,
#         'recent_avg_gf'   : rf / rt,
#         'momentum'        : mom,
#         'matches_played'  : n,
#     }

# team_stats = {}
# for t in pd.concat([recent['home_team'],
#                     recent['away_team']]).unique():
#     s = calc_stats(t, recent)
#     if s: team_stats[t] = s
# print(f"✅ Stats for {len(team_stats)} teams computed")

# # ════════════════════════════════════════════════
# # 4. HEAD-TO-HEAD RECORD
# # ════════════════════════════════════════════════
# h2h = defaultdict(lambda: {'w':0,'d':0,'l':0})
# for _, row in comp.iterrows():
#     h, a = row['home_team'], row['away_team']
#     if row['home_score'] > row['away_score']:
#         h2h[(h,a)]['w']+=1; h2h[(a,h)]['l']+=1
#     elif row['home_score'] < row['away_score']:
#         h2h[(h,a)]['l']+=1; h2h[(a,h)]['w']+=1
#     else:
#         h2h[(h,a)]['d']+=1; h2h[(a,h)]['d']+=1

# def h2h_winrate(t1, t2):
#     rec = h2h[(t1,t2)]
#     tot = rec['w'] + rec['d'] + rec['l']
#     return rec['w']/tot if tot > 0 else 0.5

# print(f"✅ H2H records computed")

# # ════════════════════════════════════════════════
# # 5. FEATURE BUILDER
# # ════════════════════════════════════════════════
# DEFL = {'win_rate':0.35,'draw_rate':0.25,'avg_gf':1.1,
#         'avg_ga':1.4,'goal_diff':-0.3,'recent_win_rate':0.35,
#         'recent_goal_diff':-0.3,'recent_avg_gf':1.0,
#         'momentum':0.4,'matches_played':20}

# def make_features(t1, t2, e1, e2):
#     s1 = team_stats.get(t1, DEFL)
#     s2 = team_stats.get(t2, DEFL)
#     hw = h2h_winrate(t1, t2)
#     return [
#         # ELO
#         e1, e2, e1-e2,
#         # Team 1 stats
#         s1['win_rate'],        s1['avg_gf'],
#         s1['avg_ga'],          s1['goal_diff'],
#         s1['recent_win_rate'], s1['recent_goal_diff'],
#         s1['recent_avg_gf'],   s1['draw_rate'],
#         s1['momentum'],
#         # Team 2 stats
#         s2['win_rate'],        s2['avg_gf'],
#         s2['avg_ga'],          s2['goal_diff'],
#         s2['recent_win_rate'], s2['recent_goal_diff'],
#         s2['recent_avg_gf'],   s2['draw_rate'],
#         s2['momentum'],
#         # Differences
#         s1['win_rate']         - s2['win_rate'],
#         s1['goal_diff']        - s2['goal_diff'],
#         s1['recent_win_rate']  - s2['recent_win_rate'],
#         s1['recent_goal_diff'] - s2['recent_goal_diff'],
#         s1['avg_gf']           - s2['avg_gf'],
#         s1['momentum']         - s2['momentum'],
#         # Head-to-head
#         hw,
#         # Home advantage (1 if t1 is "home")
#         1.0,
#     ]

# # ════════════════════════════════════════════════
# # 6. BUILD TRAINING DATA
# # (Decisive matches only — no draws → Binary)
# # ════════════════════════════════════════════════
# elo_snapshot = defaultdict(lambda: 1500.0)
# X_list, y_list, w_list = [], [], []

# for _, row in df.iterrows():
#     h, a  = row['home_team'], row['away_team']
#     e1, e2 = elo_snapshot[h], elo_snapshot[a]
#     k      = K_BASE * row['weight']
#     exp_h  = expected(e1, e2)
#     sh     = (1.0 if row['home_score'] > row['away_score'] else
#               0.0 if row['home_score'] < row['away_score'] else 0.5)

#     # Train only on 2010+ decisive competitive matches
#     decisive = (row['home_score'] != row['away_score'])
#     if row['date'].year >= 2010 and decisive:
#         X_list.append(make_features(h, a, e1, e2))
#         y_list.append(1 if row['home_score'] > row['away_score'] else 0)
#         w_list.append(row['weight'])

#     elo_snapshot[h] += k * (sh - exp_h)
#     elo_snapshot[a] += k * ((1-sh) - (1-exp_h))

# X = np.array(X_list)
# y = np.array(y_list)
# w = np.array(w_list)
# print(f"✅ Training samples : {len(X):,} decisive matches")

# # ════════════════════════════════════════════════
# # 7. TRAIN MODEL
# # ════════════════════════════════════════════════
# X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
#     X, y, w, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# Xtr    = scaler.fit_transform(X_train)
# Xte    = scaler.transform(X_test)

# print("\n⏳ Training GradientBoosting... (30-60 sec)")
# model = GradientBoostingClassifier(
#     n_estimators   = 400,
#     learning_rate  = 0.04,
#     max_depth      = 4,
#     subsample      = 0.8,
#     min_samples_leaf= 8,
#     random_state   = 42)
# model.fit(Xtr, y_train, sample_weight=w_train)
# yp = model.predict(Xte)

# print("\n════════════════════════════════")
# print("       Model Metrics")
# print("════════════════════════════════")
# print(f"Accuracy  : {accuracy_score(y_test, yp)*100:.1f}%")
# print(f"Precision : {precision_score(y_test, yp, zero_division=0)*100:.1f}%")
# print(f"Recall    : {recall_score(y_test, yp, zero_division=0)*100:.1f}%")
# print(f"F1 Score  : {f1_score(y_test, yp, zero_division=0)*100:.1f}%")

# # Cross-validation (5-fold)
# cv = cross_val_score(model, scaler.transform(X), y, cv=5)
# print(f"CV Score  : {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")
# print("(Note: R² = Regression only)")

# # ════════════════════════════════════════════════
# # 8. PREDICT FUNCTION
# # ════════════════════════════════════════════════
# ci = {c:i for i,c in enumerate(model.classes_)}

# def draw_prob(elo_diff):
#     return max(0.10, 0.27 * np.exp(-abs(elo_diff) / 400))

# def predict_match(t1, t2):
#     e1 = final_elo.get(t1, 1500)
#     e2 = final_elo.get(t2, 1500)
#     ft = scaler.transform([make_features(t1, t2, e1, e2)])
#     pr = model.predict_proba(ft)[0]
#     p1 = pr[ci.get(1, 1)]
#     p2 = pr[ci.get(0, 0)]
#     dr = draw_prob(e1 - e2)
#     p1 = p1 * (1 - dr)
#     p2 = p2 * (1 - dr)
#     tot = p1 + dr + p2
#     return p1/tot, dr/tot, p2/tot

# # ════════════════════════════════════════════════
# # 9. STRENGTH SCORE (ELO-weighted)
# # ════════════════════════════════════════════════
# def strength(t):
#     e  = final_elo.get(t, 1500)
#     s  = team_stats.get(t, DEFL)
#     en = (e - 1200) / 500
#     fm = (s['recent_win_rate']  * 0.40 +
#           s['recent_goal_diff'] * 0.25 +
#           s['win_rate']         * 0.20 +
#           s['momentum']         * 0.15)
#     return en * 0.65 + fm * 0.35

# # ════════════════════════════════════════════════
# # 10. 48 TEAMS
# # ════════════════════════════════════════════════
# teams_48 = [
#     # Europe (16)
#     'Germany','France','Spain','England','Portugal',
#     'Netherlands','Belgium','Croatia','Switzerland','Austria',
#     'Denmark','Italy','Serbia','Poland','Scotland','Turkey',
#     # South America (6)
#     'Brazil','Argentina','Uruguay','Colombia','Ecuador','Paraguay',
#     # CONCACAF (6)
#     'United States','Mexico','Canada','Costa Rica','Panama','Jamaica',
#     # Africa (9)
#     'Morocco','Senegal','Nigeria','Cameroon','Ghana',
#     'Egypt','Algeria','Ivory Coast','DR Congo',
#     # Asia (8)
#     'Japan','South Korea','Australia','Iran',
#     'Saudi Arabia','Iraq','Jordan','Uzbekistan',
#     # Oceania + Others (3)
#     'New Zealand','Venezuela','Bolivia',
# ]

# ranked_48 = sorted(teams_48, key=strength, reverse=True)

# print("\n" + "═"*62)
# print("          🏆 FIFA 2026 World Cup Prediction")
# print("═"*62)
# print("\n📊 Top 16 Strongest Teams (ELO + Form + Momentum):")
# print(f"  {'Rank':<5}{'Team':<24}{'ELO':>7}  {'Strength':>8}  Bar")
# print(f"  {'─'*60}")
# for i, t in enumerate(ranked_48[:16], 1):
#     e   = final_elo.get(t, 1500)
#     sc  = strength(t)
#     bar = "█" * max(1, int(sc * 15))
#     print(f"  {i:<5}{t:<24}{e:>7,.0f}  {sc:>8.3f}  {bar}")

# # ════════════════════════════════════════════════
# # 11. GROUP DRAW (Pot-based 12×4)
# # ════════════════════════════════════════════════
# print("\n\n📋 Group Draw — 12 Groups × 4 Teams:")
# groups = {}
# for i in range(12):
#     g = chr(65+i)
#     groups[g] = [ranked_48[i], ranked_48[i+12],
#                  ranked_48[i+24], ranked_48[i+36]]
#     print(f"  Group {g}: {' | '.join(groups[g])}")

# # ════════════════════════════════════════════════
# # 12. GROUP STAGE (Round Robin)
# # ════════════════════════════════════════════════
# print("\n\n⚽ Group Stage Simulation:")
# group_standings, third_data = {}, []

# for gname, teams in groups.items():
#     tbl = {t:{'pts':0,'gf':0,'ga':0,
#               'w':0,'d':0,'l':0} for t in teams}

#     for i in range(4):
#         for j in range(i+1, 4):
#             t1, t2      = teams[i], teams[j]
#             p1, pd_, p2 = predict_match(t1, t2)
#             if p1 > p2 and p1 > pd_:
#                 tbl[t1]['pts']+=3; tbl[t1]['w']+=1
#                 tbl[t2]['l']+=1
#                 tbl[t1]['gf']+=2;  tbl[t1]['ga']+=1
#                 tbl[t2]['gf']+=1;  tbl[t2]['ga']+=2
#             elif p2 > p1 and p2 > pd_:
#                 tbl[t2]['pts']+=3; tbl[t2]['w']+=1
#                 tbl[t1]['l']+=1
#                 tbl[t2]['gf']+=2;  tbl[t2]['ga']+=1
#                 tbl[t1]['gf']+=1;  tbl[t1]['ga']+=2
#             else:
#                 tbl[t1]['pts']+=1; tbl[t1]['d']+=1
#                 tbl[t2]['pts']+=1; tbl[t2]['d']+=1
#                 tbl[t1]['gf']+=1;  tbl[t1]['ga']+=1
#                 tbl[t2]['gf']+=1;  tbl[t2]['ga']+=1

#     st = sorted(tbl.items(),
#                 key=lambda x:(x[1]['pts'],
#                                x[1]['gf']-x[1]['ga'],
#                                x[1]['gf']),
#                 reverse=True)
#     group_standings[gname] = st

#     print(f"\n  Group {gname}:")
#     print(f"  {'Team':<22} W  D  L  GF GA  GD  Pts  Status")
#     print(f"  {'─'*60}")
#     for rk, (t, s) in enumerate(st, 1):
#         gd  = s['gf'] - s['ga']
#         tag = ("✅ Advance" if rk<=2 else
#                "〽️  3rd"   if rk==3 else "❌ Out")
#         print(f"  {t:<22} {s['w']}  {s['d']}  {s['l']}"
#               f"  {s['gf']:2}  {s['ga']:2}  {gd:+2d}  "
#               f"{s['pts']}    {tag}")

#     th = st[2]
#     third_data.append((th[0], th[1]['pts'],
#                        th[1]['gf']-th[1]['ga'],
#                        th[1]['gf'], gname))

# # ════════════════════════════════════════════════
# # 13. BEST 8 THIRD-PLACE → ROUND OF 32
# # ════════════════════════════════════════════════
# third_data.sort(key=lambda x:(x[1],x[2],x[3]), reverse=True)
# best8  = [x[0] for x in third_data[:8]]
# elim3  = [x[0] for x in third_data[8:]]

# print(f"\n\n📌 Best 8 Third-Place → Round of 32:")
# for x in third_data[:8]:
#     print(f"  ✅ {x[0]:<24} Group {x[4]}  "
#           f"Pts:{x[1]}  GD:{x[2]:+d}")
# print(f"\n  ❌ Eliminated: {', '.join(elim3)}")

# winners = [group_standings[g][0][0] for g in groups]
# runners = [group_standings[g][1][0] for g in groups]
# r32     = sorted(winners+runners+best8, key=strength, reverse=True)

# print(f"\n\n🏆 Round of 32 — 32 Teams:")
# print(f"  {'#':<4}{'Team':<24}{'ELO':>7}  {'Strength':>8}")
# print(f"  {'─'*46}")
# for i, t in enumerate(r32, 1):
#     e  = final_elo.get(t, 1500)
#     sc = strength(t)
#     print(f"  {i:<4}{t:<24}{e:>7,.0f}  {sc:>8.3f}")

# # ════════════════════════════════════════════════
# # 14. KNOCKOUT ROUNDS
# # ════════════════════════════════════════════════
# def knockout(teams, rname):
#     print(f"\n🔷 {rname}  ({len(teams)} → {len(teams)//2} teams):")
#     print(f"  {'Match':<44} Winner           Win%")
#     print(f"  {'─'*66}")
#     wl = []
#     for i in range(0, len(teams)-1, 2):
#         t1, t2       = teams[i], teams[i+1]
#         p1, pd_, p2  = predict_match(t1, t2)
#         w            = t1 if p1 >= p2 else t2
#         wp           = max(p1,p2)*100
#         m            = f"{t1} vs {t2}"
#         print(f"  {m:<44} ✅ {w:<16} {wp:.0f}%")
#         wl.append(w)
#     return wl

# cur = r32
# for rn in ["Round of 32","Round of 16",
#            "Quarter-Finals","Semi-Finals","Final"]:
#     cur = knockout(cur, rn)

# print("\n" + "═"*62)
# print(f"   🥇  PREDICTED WORLD CUP WINNER: {cur[0]} 🏆")
# print("═"*62)

# # ════════════════════════════════════════════════
# # 15. CUSTOM MATCH PREDICTOR
# # ════════════════════════════════════════════════
# print("\n\n🔮 Custom Match Predictor")
# print("─"*46)
# t1 = input("Team 1 (e.g. Brazil) : ").strip()
# t2 = input("Team 2 (e.g. France) : ").strip()
# p1, pd_, p2 = predict_match(t1, t2)
# e1 = final_elo.get(t1, 1500)
# e2 = final_elo.get(t2, 1500)
# print(f"\n  {'Team':<26} ELO      Win%")
# print(f"  {'─'*44}")
# print(f"  {t1:<26} {e1:>6,.0f}   {p1*100:.1f}%")
# print(f"  {'Draw':<26}          {pd_*100:.1f}%")
# print(f"  {t2:<26} {e2:>6,.0f}   {p2*100:.1f}%")
# print(f"\n  🏆 Predicted Winner: "
#       f"{t1 if p1>p2 else t2}")
# print("─"*46)