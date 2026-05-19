import { type ReactNode, useEffect, useState } from 'react'
import { useAuth0 } from '@auth0/auth0-react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Link, Navigate, Route, Routes } from 'react-router-dom'
import { ModelsListPage } from './pages/ModelsListPage'
import { ModelFormPage } from './pages/ModelFormPage'
import { ModelDetailPage } from './pages/ModelDetailPage'
import { RunsListPage } from './pages/RunsListPage'
import { RunDetailPage } from './pages/RunDetailPage'
import { UsersPage } from './pages/UsersPage'
import { TokenSync } from './components/TokenSync'
import { AccessPending } from './components/AccessPending'

const queryClient = new QueryClient()

function AuthButton() {
  const { logout, user } = useAuth0()
  return (
    <button
      onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}
      className="ml-auto text-gray-700 hover:text-gray-900"
    >
      {user?.email} · Logout
    </button>
  )
}

const ROLES_CLAIM = 'https://aipet/roles'

function useIsAdmin(): boolean {
  const { user } = useAuth0()
  const roles: string[] = user?.[ROLES_CLAIM] ?? []
  return roles.includes('admin')
}

function Nav() {
  const isAdmin = useIsAdmin()
  return (
    <nav className="border-b bg-white px-8 py-3 flex gap-6 text-sm font-medium items-center">
      <Link to="/models" className="text-gray-700 hover:text-gray-900">Models</Link>
      <Link to="/runs" className="text-gray-700 hover:text-gray-900">Runs</Link>
      {isAdmin && <Link to="/admin/users" className="text-gray-700 hover:text-gray-900">Users</Link>}
      <AuthButton />
    </nav>
  )
}

function AdminRoute({ children }: { children: ReactNode }) {
  const { isLoading } = useAuth0()
  const isAdmin = useIsAdmin()
  if (isLoading) return null
  return isAdmin ? <>{children}</> : <Navigate to="/models" replace />
}

function AppContent() {
  const { isAuthenticated, isLoading, loginWithRedirect, error } = useAuth0()

  const [accessDenied, setAccessDenied] = useState(false)

  useEffect(() => {
    const handler = () => setAccessDenied(true)
    window.addEventListener('auth:access-denied', handler)
    return () => window.removeEventListener('auth:access-denied', handler)
  }, [])

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      loginWithRedirect()
    }
  }, [isLoading, isAuthenticated, loginWithRedirect])

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen text-red-500">
        Authentication error: {error.message}
      </div>
    )
  }

  if (accessDenied) return <AccessPending />

  if (isLoading || !isAuthenticated) {
    return (
      <div className="flex items-center justify-center h-screen text-gray-500">
        Loading…
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <TokenSync />
      <Nav />
      <Routes>
        <Route path="/" element={<Navigate to="/models" replace />} />
        <Route path="/models" element={<ModelsListPage />} />
        <Route path="/models/new" element={<ModelFormPage />} />
        <Route path="/models/:id" element={<ModelDetailPage />} />
        <Route path="/models/:id/edit" element={<ModelFormPage />} />
        <Route path="/runs" element={<RunsListPage />} />
        <Route path="/runs/:runId" element={<RunDetailPage />} />
        <Route path="/admin/users" element={<AdminRoute><UsersPage /></AdminRoute>} />
      </Routes>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppContent />
      </BrowserRouter>
    </QueryClientProvider>
  )
}
