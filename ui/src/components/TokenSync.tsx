import { useEffect } from 'react'
import { useAuth0 } from '@auth0/auth0-react'
import { setTokenGetter } from '@/api/client'

export function TokenSync() {
  const { getAccessTokenSilently, isAuthenticated } = useAuth0()
  useEffect(() => {
    if (isAuthenticated) {
      setTokenGetter(() =>
        getAccessTokenSilently({
          authorizationParams: {
            audience: import.meta.env.VITE_AUTH0_AUDIENCE,
          },
        })
      )
    } else {
      setTokenGetter(null)
    }
  }, [isAuthenticated, getAccessTokenSilently])
  return null
}
