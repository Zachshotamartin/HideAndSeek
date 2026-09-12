"""Promotion uses a length-balanced utility and blocks any clear regression at a play length."""
import unittest

from evaluated_models import utility


def block(hider, seeker, hider_change, seeker_change, maps=8):
    return dict(maps=maps, summary={'candidate-hider': dict(hiddenFraction=hider), 'candidate-seeker': dict(hiddenFraction=seeker)},
                contrasts={'Hider change': dict(mean=sum(hider_change) / 2, bootstrap95Percent=list(hider_change)),
                           'Seeker change': dict(mean=sum(seeker_change) / 2, bootstrap95Percent=list(seeker_change))})


class RegistryTests(unittest.TestCase):
    def test_long_maps_count_once_each_and_a_long_regression_blocks_promotion(self):
        short = block(.64, .55, (.02, .12), (-.03, .05), maps=144)
        long = block(.60, .70, (.10, .30), (-.30, -.10))
        overall = block(.64, .56, (.02, .14), (-.03, .06), maps=152)
        report = dict(summary=overall['summary'], contrasts=overall['contrasts'], byPlayLength={'144': short, '750': long})
        measured = utility(report)
        self.assertAlmostEqual(measured['hiderUtility'], (.64 + .60) / 2)
        self.assertAlmostEqual(measured['seekerUtility'], ((1 - .55) + (1 - .70)) / 2)
        self.assertIn('Hider change', measured['clearImprovements'])
        self.assertIn('Seeker change at play 750', measured['roleRegressions'])
        self.assertEqual(set(measured['byPlayLength']), {'144', '750'})

    def test_report_without_length_breakdown_falls_back_to_the_overall_summary(self):
        overall = block(.6, .5, (.01, .1), (-.02, .04))
        report = dict(summary=overall['summary'], contrasts=overall['contrasts'])
        measured = utility(report)
        self.assertAlmostEqual(measured['score'], .5 * (.6 + .5))
        self.assertEqual(measured['roleRegressions'], [])


if __name__ == '__main__':
    unittest.main()
