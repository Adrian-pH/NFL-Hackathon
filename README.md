Given where all 22 players line up **before the snap**, output a full **probability distribution** over how many yards the offense will gain, not just a single number.

Conceptually we are learning a function:

f(player positions) -> p(x), x = net yards gained

Who it is for
- **Coaches** compare how an offensive formation performs against different defensive looks, and see which matchups are favorable.
- **Broadcasters** show viewers the expected gain, the spread of likely outcomes, and a confidence level, all from a pre-snap snapshot.

Analysis files:
- [formation_yardage_distribution.ipynb](formation_yardage_distribution.ipynb) — formation and label-based analysis.
- [raw_tracking_yardage_analysis.ipynb](raw_tracking_yardage_analysis.ipynb) — tracking-only analysis; no formation, coverage, PFF, or personnel labels.
