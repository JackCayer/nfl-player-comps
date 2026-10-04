Basic NFL WR Comparison, input a player's name to get the closest matches currently. 

-- Only uses 2026 stats currently 

-- Stats gathered:

    -- # of games -- # of targets -- # of receptions -- total rec yards -- total rec td's -- total rec air yards -- total YAC -- total rec epa
Comparison metrics: 

    -- Targets per Game: targets/games -- Catch rate: receptions/targets -- Yards per Target: total yards/targets -- Air yards per target: total air yards/targets -- YAC per reception: total YAC/receptions -- EPA per target: total epa/target 
-- Metrics are standardized with z-scores


Ex. comparison: Jaxon Smith-Njigba

Chris Olave 1.372396 

Garrett Wilson 2.359893 

CeeDee Lamb 2.401291 

Amon-Ra St. Brown 2.499224 

Christian Watson 2.635380 

Jalen Coker 2.800727 

Ja'Marr Chase 2.845704 

Tee Higgins 2.911925

 DeVonta Smith 2.998386 

Josh Downs 3.092949

FANTASY

-- At some points the last 8 games window contains 2025 games which are the same games used in the baseline calculation. That makes the adjusted gap smaller than it should be. Will update to use only 2026 games after week 8.

-- Players whose scoring ran well above their usage tended to score less over the following game, and those well below tended to score more. The effect was clear at the extremes but small overall, and the data covers about one season.

-- Receivers who scored well above their usage tended to score about 1 to 1.5 points less in the following game than receivers with the same recent scoring but in-line usage (2021-2026, standard scoring). Part of the overall decline in top scorers is ordinary regression to the mean, which the usage gap does not explain. Blending recent scoring with a usage-based expectation lowered prediction error by about 1%.