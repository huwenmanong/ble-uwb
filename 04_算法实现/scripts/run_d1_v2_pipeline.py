"""PyCharm entry point: independent experiment, review, and error detection."""
import os
import sys
from pathlib import Path
import shutil

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from threadpoolctl import threadpool_limits
import d1_v2_research
import d1_v2_review
import d1_v2_identifiability


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        output = d1_v2_research.main()
        if output is not None:
            for module in [d1_v2_review, d1_v2_identifiability]:
                shutil.copyfile(module.__file__, output / Path(module.__file__).name)
            d1_v2_review.main(output)
            d1_v2_identifiability.main(output)
            print(f'ALL STAGES COMPLETED: {output}', flush=True)
