Deploying the frontend to Render

1. Ensure your frontend reads the backend base URL from an env var (e.g. `REACT_APP_API_URL`).

2. In the Render dashboard when creating a new Static Site, set the build command to:

```
cd frontend && npm install && npm run build
```

and set the publish directory to `frontend/build`.

3. Add an environment variable on Render named `REACT_APP_API_URL` with the value of your backend URL (for example `https://newsnudge-backend.onrender.com`).

4. Deploy. Render will build and serve the static files via a CDN.
