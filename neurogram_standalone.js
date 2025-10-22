'use strict';

const fs = require('fs');
const path = require('path');

const LIB_DIR = path.join(__dirname, 'neurogram', 'src', 'lib');
const R = require(path.join(LIB_DIR, 'recurrent.js'));
const N = require(path.join(LIB_DIR, 'neat.js'));
const NetArt = require(path.join(LIB_DIR, 'netart.js'));

const DEFAULT_ROWS = 5;
const DEFAULT_COLS = 5;
const DEFAULT_THUMB_SIZE = 90;
const MAX_SELECTION = 4;

N.init({ nInput: 3, nOutput: 3 });

const state = {
  rows: DEFAULT_ROWS,
  cols: DEFAULT_COLS,
  thumbSize: DEFAULT_THUMB_SIZE,
  genomes: [],
  thumbs: [],
  generation: 0,
};

function createMatrix(rows, cols, factory) {
  const result = [];
  for (let r = 0; r < rows; r += 1) {
    const row = [];
    for (let c = 0; c < cols; c += 1) {
      row.push(factory(r, c));
    }
    result.push(row);
  }
  return result;
}

function resetState() {
  state.genomes = createMatrix(state.rows, state.cols, () => null);
  state.thumbs = createMatrix(state.rows, state.cols, () => null);
  state.generation = 0;
}

function randomizeRenderModeQuiet() {
  const originalLog = console.log;
  console.log = () => {};
  try {
    N.randomizeRenderMode();
  } finally {
    console.log = originalLog;
  }
}

function initGenome() {
  randomizeRenderModeQuiet();
  for (let r = 0; r < state.rows; r += 1) {
    for (let c = 0; c < state.cols; c += 1) {
      state.genomes[r][c] = new N.Genome();
    }
  }

  for (let k = 0; k < 8; k += 1) {
    for (let r = 0; r < state.rows; r += 1) {
      for (let c = 0; c < state.cols; c += 1) {
        if (Math.random() < 0.5) state.genomes[r][c].addRandomNode();
        if (Math.random() < 0.5) state.genomes[r][c].addRandomConnection();
      }
    }
  }
}

function generateThumbnail(row, col) {
  const genome = state.genomes[row][col];
  genome.roundWeights();
  state.thumbs[row][col] = NetArt.genGenomeImage(genome, state.thumbSize, state.thumbSize);
}

function initThumbs() {
  for (let r = 0; r < state.rows; r += 1) {
    for (let c = 0; c < state.cols; c += 1) {
      generateThumbnail(r, c);
    }
  }
  state.generation = 1;
}

function indexToCoord(index) {
  const row = Math.floor(index / state.cols);
  const col = index % state.cols;
  return { row, col };
}

function coordToIndex(row, col) {
  return row * state.cols + col;
}

function validateSelection(selected) {
  if (!Array.isArray(selected)) {
    return { ok: false, error: 'selected must be an array' };
  }
  if (selected.length === 0) {
    return { ok: false, error: 'selected cannot be empty' };
  }
  const maxIndex = state.rows * state.cols - 1;
  const unique = new Set();
  for (const value of selected) {
    if (typeof value !== 'number' || !Number.isInteger(value)) {
      return { ok: false, error: 'selection indices must be integers' };
    }
    if (value < 0 || value > maxIndex) {
      return { ok: false, error: `selection index ${value} out of range` };
    }
    unique.add(value);
  }
  if (unique.size > MAX_SELECTION) {
    return { ok: false, error: `at most ${MAX_SELECTION} selections allowed` };
  }
  return { ok: true, selected: Array.from(unique) };
}

function encodeImage(image) {
  const height = image.n;
  const width = image.d;
  const total = width * height;
  const buffer = Buffer.alloc(total * 3);

  for (let i = 0; i < total; i += 1) {
    const base = i * 3;
    buffer[base] = Math.max(0, Math.min(255, Math.round(image.r[i] * 255)));
    buffer[base + 1] = Math.max(0, Math.min(255, Math.round(image.g[i] * 255)));
    buffer[base + 2] = Math.max(0, Math.min(255, Math.round(image.b[i] * 255)));
  }

  return {
    width,
    height,
    data: buffer.toString('base64'),
  };
}

function buildStatePayload() {
  const images = [];
  for (let r = 0; r < state.rows; r += 1) {
    for (let c = 0; c < state.cols; c += 1) {
      images.push({
        index: coordToIndex(r, c),
        row: r,
        col: c,
        ...encodeImage(state.thumbs[r][c]),
      });
    }
  }

  return {
    generation: state.generation,
    rows: state.rows,
    cols: state.cols,
    thumbSize: state.thumbSize,
    renderMode: N.getRenderMode(),
    images,
  };
}

function buildGenomeSnapshot() {
  const genomes = [];
  for (let r = 0; r < state.rows; r += 1) {
    const row = [];
    for (let c = 0; c < state.cols; c += 1) {
      row.push(state.genomes[r][c].toJSON());
    }
    genomes.push(row);
  }
  return genomes;
}

function buildSaveState(options = {}) {
  const includeImages = options.includeImages || false;
  const snapshot = {
    generation: state.generation,
    rows: state.rows,
    cols: state.cols,
    thumbSize: state.thumbSize,
    renderMode: N.getRenderMode(),
    genomes: buildGenomeSnapshot(),
  };
  if (includeImages) {
    snapshot.images = buildStatePayload().images;
  }
  return snapshot;
}

function ensureInitialized() {
  if (state.generation === 0) {
    throw new Error('Population not initialized. Call initPop first.');
  }
}

function evolve(selected) {
  ensureInitialized();
  const validated = validateSelection(selected);
  if (!validated.ok) {
    throw new Error(validated.error);
  }

  const picks = validated.selected;
  const preserveSet = new Set(picks);

  const chooseParent = () => {
    const ix = R.randi(0, picks.length);
    return picks[ix];
  };

  const total = state.rows * state.cols;
  for (let index = 0; index < total; index += 1) {
    if (preserveSet.has(index)) {
      continue;
    }
    const momIndex = chooseParent();
    const dadIndex = chooseParent();

    const { row: momRow, col: momCol } = indexToCoord(momIndex);
    const { row: dadRow, col: dadCol } = indexToCoord(dadIndex);
    const { row, col } = indexToCoord(index);

    const momGenome = state.genomes[momRow][momCol];
    const dadGenome = state.genomes[dadRow][dadCol];

    if (momIndex === dadIndex) {
      state.genomes[row][col] = momGenome.copy();
    } else {
      state.genomes[row][col] = momGenome.crossover(dadGenome);
    }

    const genome = state.genomes[row][col];
    genome.mutateWeights();
    if (Math.random() < 0.5) genome.addRandomNode();
    if (Math.random() < 0.5) genome.addRandomConnection();
    genome.roundWeights();
    state.thumbs[row][col] = NetArt.genGenomeImage(genome, state.thumbSize, state.thumbSize);
  }

  state.generation += 1;
  return buildStatePayload();
}

function loadState(snapshot) {
  if (!snapshot || typeof snapshot !== 'object') {
    throw new Error('snapshot must be an object');
  }
  const { rows, cols, thumbSize, generation, renderMode, genomes } = snapshot;
  if (!Number.isInteger(rows) || rows <= 0) {
    throw new Error('snapshot.rows must be a positive integer');
  }
  if (!Number.isInteger(cols) || cols <= 0) {
    throw new Error('snapshot.cols must be a positive integer');
  }
  if (!Number.isInteger(thumbSize) || thumbSize <= 0) {
    throw new Error('snapshot.thumbSize must be a positive integer');
  }
  if (!Number.isInteger(generation) || generation < 1) {
    throw new Error('snapshot.generation must be a positive integer');
  }
  if (!Array.isArray(genomes) || genomes.length !== rows) {
    throw new Error('snapshot.genomes has unexpected shape');
  }

  state.rows = rows;
  state.cols = cols;
  state.thumbSize = thumbSize;
  resetState();
  state.generation = generation;
  if (typeof renderMode === 'number' && typeof N.setRenderMode === 'function') {
    N.setRenderMode(renderMode);
  }

  for (let r = 0; r < rows; r += 1) {
    if (!Array.isArray(genomes[r]) || genomes[r].length !== cols) {
      throw new Error('snapshot.genomes has unexpected shape');
    }
    for (let c = 0; c < cols; c += 1) {
      const genomeData = genomes[r][c];
      const genome = new N.Genome();
      genome.fromJSON(genomeData);
      state.genomes[r][c] = genome;
      state.thumbs[r][c] = NetArt.genGenomeImage(genome, state.thumbSize, state.thumbSize);
    }
  }

  return buildStatePayload();
}

function initPop(options = {}) {
  const { rows, cols, thumbSize } = options;
  if (rows !== undefined) {
    if (!Number.isInteger(rows) || rows <= 0) {
      throw new Error('rows must be a positive integer');
    }
    state.rows = rows;
  }
  if (cols !== undefined) {
    if (!Number.isInteger(cols) || cols <= 0) {
      throw new Error('cols must be a positive integer');
    }
    state.cols = cols;
  }
  if (thumbSize !== undefined) {
    if (!Number.isInteger(thumbSize) || thumbSize <= 0) {
      throw new Error('thumbSize must be a positive integer');
    }
    state.thumbSize = thumbSize;
  }

  resetState();
  initGenome();
  initThumbs();
  return buildStatePayload();
}

function getState() {
  ensureInitialized();
  return buildStatePayload();
}

function getImage(index) {
  ensureInitialized();
  const maxIndex = state.rows * state.cols - 1;
  if (!Number.isInteger(index) || index < 0 || index > maxIndex) {
    throw new Error(`index must be between 0 and ${maxIndex}`);
  }
  const { row, col } = indexToCoord(index);
  return {
    index,
    row,
    col,
    ...encodeImage(state.thumbs[row][col]),
  };
}

function dumpState(pathLike) {
  ensureInitialized();
  const target = path.resolve(process.cwd(), pathLike);
  fs.writeFileSync(target, JSON.stringify(buildSaveState(true), null, 2), 'utf8');
  return target;
}

function exportState(options) {
  ensureInitialized();
  return buildSaveState(options || {});
}

module.exports = {
  initPop,
  getState,
  evolve,
  getImage,
  dumpState,
  exportState,
  loadState,
  MAX_SELECTION,
};
