import random


class SelfShuffleList(object):
    def __init__(self, li):
        assert (type(li) is list and li)  # cannot be empty
        self.li = li  # be careful! not a deep copy!
        self.idx = 0
        self._shuffle()  # shuffle at the beginning

    def get_next(self):
        if self.idx < len(self.li):
            rtn = self.li[self.idx]
            self.idx += 1
            return rtn
        else:
            self.idx = 0
            self._shuffle()
            return self.li[self.idx]

    def get_next_n(self, n):
        if n > len(self.li):
            raise RuntimeError('Cannot sample {} elements of an {} list'.format(
                n, len(self.li)))
        end_idx = self.idx + n
        if end_idx < len(self.li):
            rtn = self.li[self.idx:end_idx]
            self.idx = end_idx
            return rtn
        else:
            rtn = self.li[self.idx:]
            # self._shuffle()  # shouldn't shuffle since may have duplicate at the end
            can_supply = len(self.li) - self.idx
            new_idx = n - can_supply
            rtn += self.li[0:new_idx]
            self.idx = new_idx
            assert (self.idx >= 0 and self.idx < len(self.li))
            self._shuffle()
            return rtn

    def _shuffle(self):
        random.Random(123).shuffle(self.li)


if __name__ == '__main__':
    sl = SelfShuffleList(list(range(10)))
    for i in range(10):
        print(i, sl.get_next_n(4))
