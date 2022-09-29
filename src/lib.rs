//! # Extended Isolation Forest
//!
//! This is a rust port of the anomaly detection algorithm described in [Extended Isolation Forest](https://doi.org/10.1109/TKDE.2019.2947676)
//! and implemented in [https://github.com/sahandha/eif](https://github.com/sahandha/eif). For a detailed description see the paper or the
//! github repository.
//!
//! This crate requires rust >= 1.51 as it makes use of `min_const_generics`.
//!
//! Includes optional serde support with the `serde` feature.
//!
//! ## Example
//!
//! ```rust
//! use rand::distributions::Uniform;
//! use rand::Rng;
//! use extended_isolation_forest::{Forest, ForestOptions};
//!
//! fn make_f64_forest() -> Forest<f64, 3> {
//!     let rng = &mut rand::thread_rng();
//!     let distribution = Uniform::new(-4., 4.);
//!     let distribution2 = Uniform::new(10., 50.);
//!     let values: Vec<_> = (0..3000)
//!         .map(|_| [rng.sample(distribution), rng.sample(distribution), rng.sample(distribution2)])
//!         .collect();
//!
//!     let options = ForestOptions {
//!         n_trees: 150,
//!         sample_size: 200,
//!         max_tree_depth: None,
//!         extension_level: 1,
//!     };
//!     Forest::from_slice(values.as_slice(), &options).unwrap()
//! }
//!
//! fn main() {
//!     let forest = make_f64_forest();
//!
//!     // no anomaly
//!     assert!(forest.score(&[1.0, 3.0, 25.0]) < 0.5);
//!     assert!(forest.score(&[-1.0, 3.0, 25.0]) < 0.5);
//!
//!     // anomalies
//!     assert!(forest.score(&[-12.0, 6.0, 25.0]) > 0.5);
//!     assert!(forest.score(&[-1.0, 2.0, 60.0]) > 0.5);
//!     assert!(forest.score(&[-1.0, 2.0, 0.0]) > 0.5);
//! }
//! ```

use std::ops::Index;
use std::result::Result;
use std::{boxed::Box, fmt::Debug};

use num::{traits::FloatConst, Float};
use plotly::layout::{Shape, ShapeLayer, ShapeType};
use rand::{
    distributions::{uniform::SampleUniform, Uniform},
    rngs::ThreadRng,
    seq::{IteratorRandom, SliceRandom},
    Rng,
};
use rand_distr::{Distribution, StandardNormal};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use crate::error::Error;

mod error;
#[cfg(feature = "serde")]
mod serde_array;

#[cfg(not(feature = "serde"))]
pub trait ForestFloat<'de>: Float + Debug {}

#[cfg(feature = "serde")]
pub trait ForestFloat<'de>: Float + Serialize + Deserialize<'de> {}

impl<'de> ForestFloat<'de> for f32 {}
impl<'de> ForestFloat<'de> for f64 {}

pub struct ForestOptions {
    /// `n_trees` is the number of trees to be created.
    pub n_trees: usize,

    /// `sample_size` is the number of samples of the training data to be used in
    /// creation of each tree. Must be smaller than `training_data.len()`.
    pub sample_size: usize,

    /// `max_tree_depth` is the max. allowed tree depth. This is by default set to average
    /// length of an unsuccessful search in a binary tree.
    pub max_tree_depth: Option<usize>,

    /// `extension_level` specifies degree of freedom in choosing the hyperplanes for dividing up
    /// data. Must be smaller than the dimension n of the dataset.
    pub extension_level: usize,
}

impl Default for ForestOptions {
    fn default() -> Self {
        Self {
            n_trees: 20,
            sample_size: 20,
            max_tree_depth: None,
            extension_level: 0,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Forest<T, const N: usize> {
    /// Multiplicative factor used in computing the anomaly scores.
    avg_path_length_c: f64,

    trees: Vec<Tree<T, N>>,
}

impl<'de, T, const N: usize> Forest<T, N>
where
    T: ForestFloat<'de> + SampleUniform + Default,
    StandardNormal: Distribution<T>,
{
    /// Build a new forest from the given training data
    pub fn from_slice(training_data: &[[T; N]], options: &ForestOptions) -> Result<Self, Error> {
        if training_data.len() < options.sample_size || N == 0 {
            return Err(Error::InsufficientTrainingData);
        } else if options.extension_level > (N - 1) {
            return Err(Error::ExtensionLevelExceedsDimensions);
        }

        let max_tree_depth = if let Some(mdt) = options.max_tree_depth {
            mdt
        } else {
            (options.sample_size as f64).log2().ceil() as usize
        };

        // build the trees
        let rng = &mut rand::thread_rng();
        let trees = (0..options.n_trees)
            .map(|_| {
                let tree_sample: Vec<_> = training_data
                    .choose_multiple(rng, options.sample_size)
                    .into_iter()
                    .collect();

                Tree::new(
                    tree_sample.as_slice(),
                    rng,
                    max_tree_depth,
                    options.extension_level,
                )
            })
            .collect();

        Ok(Self {
            avg_path_length_c: c_factor(options.sample_size),
            trees,
        })
    }

    /// compute anomaly score for an item
    pub fn score(&self, values: &[T; N]) -> f64 {
        let path_length: f64 = self.trees.iter().map(|tree| tree.path_length(values)).sum();

        // Average of path length travelled by the point in all trees.
        let eh = path_length / self.trees.len() as f64;

        // Anomaly Score
        2.0_f64.powf(-eh / self.avg_path_length_c)
    }

    pub fn add_tree_splits(&self, idx: usize, layout: &mut plotly::Layout) {
        self.trees[idx].add_splits(layout);
    }
}

impl<T, const N: usize> Index<usize> for Forest<T, N> {
    type Output = Tree<T, N>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.trees[index]
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
enum Node<T, const N: usize> {
    Ex(ExNode),
    In(InNode<T, N>),
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct InNode<T, const N: usize> {
    /// Left child node.
    left: Box<Node<T, N>>,

    /// Right child node.
    right: Box<Node<T, N>>,

    /// Normal vector at the root of this tree, which is used in
    /// creating hyperplanes for splitting criteria
    #[cfg_attr(feature = "serde", serde(with = "serde_array"))]
    n: [T; N],

    /// Intercept point through which the hyperplane passes.
    #[cfg_attr(feature = "serde", serde(with = "serde_array"))]
    p: [T; N],
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct ExNode {
    /// Size of the dataset present at the node.
    num_samples: usize,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Tree<T, const N: usize> {
    root: Node<T, N>,
}

impl<'de, T, const N: usize> Tree<T, N>
where
    T: ForestFloat<'de> + SampleUniform + Default,
    StandardNormal: Distribution<T>,
{
    pub fn new(
        samples: &[&[T; N]],
        rng: &mut ThreadRng,
        max_tree_depth: usize,
        extension_level: usize,
    ) -> Self {
        Self {
            root: Self::make_tree(samples, rng, 0, max_tree_depth, extension_level),
        }
    }

    fn make_tree(
        samples: &[&[T; N]],
        rng: &mut ThreadRng,
        current_tree_depth: usize,
        max_tree_depth: usize,
        extension_level: usize,
    ) -> Node<T, N> {
        let num_samples = samples.len();
        if current_tree_depth >= max_tree_depth || num_samples <= 1 {
            Node::Ex(ExNode { num_samples })
        } else {
            // randomly select an intercept point p ~ ∈ IR |samples| in
            // the range of the samples
            let p = {
                let mut maxs = *samples[0];
                let mut mins = *samples[0];
                samples.iter().skip(1).for_each(|s| {
                    s.iter().enumerate().for_each(|(i, v)| {
                        maxs[i] = if *v > maxs[i] { *v } else { maxs[i] };
                        mins[i] = if *v < mins[i] { *v } else { mins[i] };
                    })
                });

                // randomly pick an intercept point using a uniform distribution
                let mut p = [T::zero(); N];
                mins.iter().zip(maxs.iter()).zip(p.iter_mut()).for_each(
                    |((min_val, max_val), p_i)| {
                        *p_i = if min_val == max_val {
                            *min_val
                        } else {
                            rng.sample(Uniform::new(min_val, max_val))
                        }
                    },
                );
                p
            };

            // randomly select a normal vector ~n ∈ IR |samples| by drawing each coordinate
            // of ~n from a standard Gaussian distribution.
            let mut n = [T::zero(); N];
            (0..N)
                .zip(n.iter_mut())
                .for_each(|(_, n_i)| *n_i = rng.sample(StandardNormal));

            // set coordinates of ~n to zero according to extension level
            for idx in (0..N).choose_multiple(rng, N - extension_level - 1) {
                n[idx] = T::zero();
            }

            let mut samples_left = vec![];
            let mut samples_right = vec![];

            for sample in samples {
                match determinate_direction(*sample, &n, &p) {
                    Direction::Left => samples_left.push(*sample),
                    Direction::Right => samples_right.push(*sample),
                }
            }

            Node::In(InNode {
                left: Box::new(Self::make_tree(
                    samples_left.as_slice(),
                    rng,
                    current_tree_depth + 1,
                    max_tree_depth,
                    extension_level,
                )),
                right: Box::new(Self::make_tree(
                    samples_right.as_slice(),
                    rng,
                    current_tree_depth + 1,
                    max_tree_depth,
                    extension_level,
                )),
                n,
                p,
            })
        }
    }

    /// length of the path traversed by the point on the tree when it reaches an external node.
    pub fn path_length(&self, values: &[T; N]) -> f64 {
        self.path_length_recurse(&self.root, values)
    }

    fn path_length_recurse(&self, node: &Node<T, N>, values: &[T; N]) -> f64 {
        match node {
            Node::Ex(ex_node) => {
                if ex_node.num_samples <= 1 {
                    0.0
                } else {
                    c_factor(ex_node.num_samples)
                }
            }
            Node::In(in_node) => {
                1.0 + self.path_length_recurse(
                    match determinate_direction(values, &in_node.n, &in_node.p) {
                        Direction::Left => in_node.left.as_ref(),
                        Direction::Right => in_node.right.as_ref(),
                    },
                    values,
                )
            }
        }
    }

    fn get_splits(&self) -> Vec<([T; N], [T; N])> {
        NodeIter::new(self)
            .filter_map(|node| match node {
                Node::Ex(_ex_node) => None,
                Node::In(in_node) => Some((in_node.p, in_node.n)),
            })
            .collect()
    }

    // // https://en.wikipedia.org/wiki/Line–line_intersection
    // fn intersection(split1: ([T; N], [T; N]), split2: ([T; N], [T; N])) -> Option<(T, T)> {
    //     let (x1, y1) = (split1.0[0], split1.0[1]);
    //     let (x2, y2) = (split1.1[0], split1.1[1]);
    //     let (x3, y3) = (split2.0[0], split2.0[1]);
    //     let (x4, y4) = (split2.1[0], split2.1[1]);
    //     let px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4))
    //         / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
    //     let py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4))
    //         / ((x1 - x2) * (y3 - y4) * (y1 - y2) * (x3 - x4));
    //     Some((px, py))
    // }

    pub fn add_splits(&self, layout: &mut plotly::Layout) {
        let mut lines = Vec::new();

        let splits = self.get_splits();
        for (i, (pos, norm)) in splits.iter().enumerate() {
            // horizontal
            if norm[0].abs() > T::zero() {
                let y = pos[1];
                let x_left = splits[0..i]
                    .iter()
                    .filter(|(p, n)| n[1].abs() > T::zero() && p[0] <= pos[0])
                    .map(|(p, _n)| p[0])
                    .fold(T::zero(), T::max);
                let x_right = splits[0..i]
                    .iter()
                    .filter(|(p, n)| n[1].abs() > T::zero() && p[0] >= pos[0])
                    .map(|(p, _n)| p[0])
                    .fold(T::one(), T::min);
                lines.push(([x_left, y], [x_right, y]))
            }
            // vertical
            if norm[1].abs() > T::zero() {
                let x = pos[0];
                let y_down = splits[0..i]
                    .iter()
                    .filter(|(p, n)| n[0].abs() > T::zero() && p[1] >= pos[1])
                    .map(|(p, _n)| p[1])
                    .fold(T::zero(), T::max);
                let y_up = splits[0..i]
                    .iter()
                    .filter(|(p, n)| n[0].abs() > T::zero() && p[1] <= pos[1])
                    .map(|(p, _n)| p[1])
                    .fold(T::one(), T::min);
                lines.push(([x, y_down], [x, y_up]))
            }
        }

        for ([x0, y0], [x1, y1]) in lines {
            layout.add_shape(
                Shape::new()
                    .shape_type(ShapeType::Line)
                    .layer(ShapeLayer::Below)
                    .x0(x0.to_f64().unwrap())
                    .y0(y0.to_f64().unwrap())
                    .x1(x1.to_f64().unwrap())
                    .y1(y1.to_f64().unwrap()),
            );
        }
    }
}

struct NodeIter<'a, T, const N: usize> {
    deque: std::collections::VecDeque<&'a Node<T, N>>,
}

impl<'a, T, const N: usize> NodeIter<'a, T, N> {
    fn new(tree: &'a Tree<T, N>) -> Self {
        let mut deque = std::collections::VecDeque::new();
        deque.push_back(&tree.root);
        Self { deque }
    }
}

impl<'a, T, const N: usize> Iterator for NodeIter<'a, T, N> {
    type Item = &'a Node<T, N>;

    fn next(&mut self) -> Option<Self::Item> {
        self.deque.pop_front().map(|node| {
            match node {
                Node::Ex(_ex_node) => {}
                Node::In(in_node) => {
                    self.deque.push_back(&in_node.left);
                    self.deque.push_back(&in_node.right);
                }
            }
            node
        })
    }
}

/// Average path length of unsuccessful search in a binary search tree given n points
/// n: Number of data points for the BST.
///
/// Returns the average path length of unsuccessful search in a BST
fn c_factor(n: usize) -> f64 {
    2.0 * ((n as f64 - 1.0).log(f64::E()) + 0.5772156649) - (2.0 * (n as f64 - 1.0) / n as f64)
}

enum Direction {
    Left,
    Right,
}

fn determinate_direction<T, const N: usize>(sample: &[T; N], n: &[T; N], p: &[T; N]) -> Direction
where
    T: Float,
{
    let direction_value = sample
        .iter()
        .zip(p.iter())
        .map(|(sample_val, p_val)| *sample_val - *p_val)
        .zip(n.iter())
        .fold(T::zero(), |sum, (sp_val, n_val)| sum + sp_val * (*n_val));

    if direction_value <= T::zero() {
        Direction::Left
    } else {
        Direction::Right
    }
}

#[cfg(test)]
mod tests {
    use rand::distributions::Uniform;
    use rand::Rng;

    use crate::{Forest, ForestOptions};

    fn make_f64_forest() -> Forest<f64, 3> {
        let rng = &mut rand::thread_rng();
        let distribution = Uniform::new(-4., 4.);
        let distribution2 = Uniform::new(10., 50.);

        let values: Vec<_> = (0..6000)
            .map(|_| {
                [
                    rng.sample(distribution),
                    rng.sample(distribution),
                    rng.sample(distribution2),
                ]
            })
            .collect();

        let options = ForestOptions {
            n_trees: 150,
            sample_size: 200,
            max_tree_depth: None,
            extension_level: 1,
        };
        Forest::from_slice(values.as_slice(), &options).unwrap()
    }

    fn assert_anomalies_forest_3d_f64(forest: &Forest<f64, 3>) {
        // no anomaly
        assert!(forest.score(&[1.0, 3.0, 25.0]) < 0.5);
        assert!(forest.score(&[-1.0, 3.0, 25.0]) < 0.5);

        // anomalies
        assert!(forest.score(&[-12.0, 6.0, 25.0]) > 0.5);
        assert!(forest.score(&[-1.0, 2.0, 60.0]) > 0.5);
        assert!(forest.score(&[-1.0, 2.0, 0.0]) > 0.5);
    }

    #[test]
    fn score_forest_3d_f64() {
        let forest = make_f64_forest();
        assert_anomalies_forest_3d_f64(&forest);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialize_forest_3d_f64() {
        let forest = make_f64_forest();
        let forest_json = serde_json::to_string(&forest).unwrap();
        let forest2 = serde_json::from_str(forest_json.as_str()).unwrap();
        assert_anomalies_forest_3d_f64(&forest2);
    }
}
