import sys
from contextlib import contextmanager

class NullOutput:
    def write(self, s):
        pass

@contextmanager
def suppress_output(suppress: bool = True):
    if suppress:
        # Save the current stdout and stderr streams
        stdout, stderr = sys.stdout, sys.stderr

        # Replace stdout and stderr with NullOutput objects that discard any output
        sys.stdout, sys.stderr = NullOutput(), NullOutput()

        try:
            # Yield control to the code block
            yield
        finally:
            # Restore the original stdout and stderr streams
            sys.stdout, sys.stderr = stdout, stderr
    else:
        yield
