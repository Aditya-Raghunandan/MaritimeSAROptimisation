import { defineConfig } from 'vite';

/*
  BASE PATH. GitHub Pages serves a project site from
  https://<user>.github.io/<repo>/, not from the domain root, so every asset
  URL has to be prefixed with the repo name. Without this the built page loads
  and then fetches /assets/index.js -- which is a 404 at the domain root -- and
  you get a blank screen with no error on the page itself.

  It is set from an environment variable rather than hard-coded so `npm run
  dev` and `npm run preview` keep working at "/" locally. The deploy workflow
  sets VITE_BASE.
*/
export default defineConfig({
  base: process.env.VITE_BASE ?? '/',
  build: {
    // The codec wasm that zarrita loads on demand is already a separate chunk;
    // this only silences the size warning for it, which is expected and not a
    // problem because it is fetched lazily and only when a store is opened.
    chunkSizeWarningLimit: 900,
  },
  test: {
    // Tests live in frontend/tests/, moved there on review feedback.
    include: ['tests/**/*.test.js'],
  },
});
