def normalize_odds(odds1, odds2):
    pw, pl = 1./odds1, 1./odds2
    pw, pl = pw/(pw+pl), pl/(pw+pl)
    odds1, odds2 = 1./pw, 1./pl
    return odds1, odds2
