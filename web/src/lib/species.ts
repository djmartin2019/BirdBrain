import speciesData from './species.json';

/** Merged CUB-200 + iNaturalist bird species (deduplicated display names). */
export const species: readonly string[] = speciesData;

export const SPECIES_COUNT = species.length;

/** Same list sorted alphabetically for display. */
export const speciesAlphabetical: readonly string[] = [...species].sort((a, b) =>
	a.localeCompare(b)
);
