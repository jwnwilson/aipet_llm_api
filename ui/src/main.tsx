import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Auth0Provider } from '@auth0/auth0-react'
import './index.css'
import App from './App.tsx'

const _domain = import.meta.env.VITE_AUTH0_DOMAIN
const _clientId = import.meta.env.VITE_AUTH0_CLIENT_ID
if (!_domain || !_clientId) {
  throw new Error('Missing VITE_AUTH0_DOMAIN or VITE_AUTH0_CLIENT_ID — copy .env.local.example to .env.local')
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Auth0Provider
      domain={import.meta.env.VITE_AUTH0_DOMAIN}
      clientId={import.meta.env.VITE_AUTH0_CLIENT_ID}
      authorizationParams={{
        redirect_uri: window.location.origin,
        audience: import.meta.env.VITE_AUTH0_AUDIENCE,
      }}
    >
      <App />
    </Auth0Provider>
  </StrictMode>,
)
