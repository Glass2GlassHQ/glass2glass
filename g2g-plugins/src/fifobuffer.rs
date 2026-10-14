use alloc::vec::Vec;

// items read from the front through an offset, so a take does not move what is left
#[derive(Debug, Clone)]
pub(crate) struct FifoBuffer<T> {
    items: Vec<T>,
    read_offset: usize,
}

impl<T> Default for FifoBuffer<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            read_offset: 0,
        }
    }
}

impl<T: Copy> FifoBuffer<T> {
    pub(crate) fn len(&self) -> usize {
        self.items.len() - self.read_offset
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn clear(&mut self) {
        self.items.clear();
        self.read_offset = 0;
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.items[self.read_offset..]
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.items[self.read_offset..]
    }

    pub(crate) fn extend_from_slice(&mut self, items: &[T]) {
        self.compact();
        self.items.extend_from_slice(items);
    }

    pub(crate) fn resize(&mut self, len: usize, fill: T) {
        self.compact();
        self.items.resize(self.read_offset + len, fill);
    }

    pub(crate) fn prepend(&mut self, count: usize, fill: T) {
        let mut grown = Vec::with_capacity(count + self.len());
        grown.resize(count, fill);
        grown.extend_from_slice(self.as_slice());
        self.items = grown;
        self.read_offset = 0;
    }

    pub(crate) fn take_front(&mut self, count: usize) -> Vec<T> {
        let taken = self.as_slice()[..count].to_vec();
        self.discard(count);
        taken
    }

    pub(crate) fn discard(&mut self, count: usize) {
        self.read_offset += count.min(self.len());
        if self.read_offset == self.items.len() {
            self.clear();
        }
    }

    // the consumed prefix pays for the move, which keeps a take O(1) amortized
    fn compact(&mut self) {
        if self.read_offset > 0 && self.read_offset >= self.len() {
            self.items.drain(..self.read_offset);
            self.read_offset = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn takes_come_out_in_push_order_across_compactions() {
        let mut fifo = FifoBuffer::default();
        let mut next = 0u32;
        let mut expected = 0u32;
        for round in 0..50u32 {
            let pushed: Vec<u32> = (next..next + round % 7 + 1).collect();
            next += pushed.len() as u32;
            fifo.extend_from_slice(&pushed);
            let take = fifo.len().min((round % 5) as usize + 1);
            for value in fifo.take_front(take) {
                assert_eq!(value, expected);
                expected += 1;
            }
        }
        assert_eq!(fifo.len(), (next - expected) as usize);
        assert_eq!(fifo.as_slice().first().copied(), Some(expected));
    }

    #[test]
    fn resize_and_prepend_work_on_the_unread_part() {
        let mut fifo = FifoBuffer::default();
        fifo.extend_from_slice(&[1u8, 2, 3, 4]);
        fifo.discard(2);
        fifo.resize(4, 0);
        assert_eq!(fifo.as_slice(), [3, 4, 0, 0]);
        fifo.prepend(1, 9);
        assert_eq!(fifo.as_slice(), [9, 3, 4, 0, 0]);
        fifo.as_mut_slice()[0] = 8;
        assert_eq!(fifo.take_front(2), [8, 3]);
        fifo.discard(10);
        assert!(fifo.is_empty());
    }
}
