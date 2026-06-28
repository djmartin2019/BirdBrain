import speciesData from './species.json';

/** All 200 CUB-200-2011 species the models were trained on (dataset class order). */
export const species: readonly string[] = speciesData;

export const SPECIES_COUNT = species.length;

/** Same list sorted alphabetically for display. */
export const speciesAlphabetical: readonly string[] = [...species].sort((a, b) =>
	a.localeCompare(b)
);
