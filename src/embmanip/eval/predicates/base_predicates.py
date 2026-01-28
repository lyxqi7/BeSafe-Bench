class FinOn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_ontop(arg1)

class NotOn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return not arg2.check_on(arg1)

