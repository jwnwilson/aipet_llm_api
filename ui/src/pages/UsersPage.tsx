import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { UserCheck, UserX } from 'lucide-react'
import { approveUser, listUsers, revokeUser } from '@/api/admin'
import { Button } from '@/components/ui/button'
import type { UserContext } from '@/types'

export function UsersPage() {
  const queryClient = useQueryClient()

  const { data: pending = [], isLoading: loadingPending } = useQuery({
    queryKey: ['users', 'pending'],
    queryFn: () => listUsers('pending'),
  })

  const { data: approved = [], isLoading: loadingApproved } = useQuery({
    queryKey: ['users', 'approved'],
    queryFn: () => listUsers('approved'),
  })

  const approveMutation = useMutation({
    mutationFn: (user: UserContext) => approveUser(user.user_id, user.email),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', 'pending'] })
      queryClient.invalidateQueries({ queryKey: ['users', 'approved'] })
    },
  })

  const revokeMutation = useMutation({
    mutationFn: (user_id: string) => revokeUser(user_id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', 'pending'] })
      queryClient.invalidateQueries({ queryKey: ['users', 'approved'] })
    },
  })

  return (
    <div className="p-8 space-y-10">
      <section>
        <h2 className="text-xl font-semibold mb-4">Awaiting Approval</h2>
        {loadingPending ? (
          <p className="text-gray-500">Loading…</p>
        ) : (
          <div className="rounded-md border bg-white overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-gray-50 text-gray-500 text-xs uppercase tracking-wide">
                  <th className="text-left px-4 py-3 font-semibold">Email</th>
                  <th className="text-left px-4 py-3 font-semibold">User ID</th>
                  <th className="text-left px-4 py-3 font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {pending.length === 0 ? (
                  <tr>
                    <td colSpan={3} className="text-center py-8 text-gray-400">
                      No users awaiting approval
                    </td>
                  </tr>
                ) : (
                  pending.map(user => (
                    <tr key={user.user_id} className="border-b last:border-0 hover:bg-gray-50">
                      <td className="px-4 py-3 text-gray-700">{user.email ?? '—'}</td>
                      <td className="px-4 py-3 font-mono text-gray-700 text-xs">{user.user_id}</td>
                      <td className="px-4 py-3">
                        <Button
                          size="sm"
                          onClick={() => approveMutation.mutate(user)}
                          disabled={approveMutation.isPending}
                          aria-label={`Approve ${user.email ?? user.user_id}`}
                        >
                          <UserCheck className="h-3.5 w-3.5 mr-1" />Approve
                        </Button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-4">Approved Users</h2>
        {loadingApproved ? (
          <p className="text-gray-500">Loading…</p>
        ) : (
          <div className="rounded-md border bg-white overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-gray-50 text-gray-500 text-xs uppercase tracking-wide">
                  <th className="text-left px-4 py-3 font-semibold">Email</th>
                  <th className="text-left px-4 py-3 font-semibold">User ID</th>
                  <th className="text-left px-4 py-3 font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {approved.length === 0 ? (
                  <tr>
                    <td colSpan={3} className="text-center py-8 text-gray-400">
                      No approved users
                    </td>
                  </tr>
                ) : (
                  approved.map(user => (
                    <tr key={user.user_id} className="border-b last:border-0 hover:bg-gray-50">
                      <td className="px-4 py-3 text-gray-700">{user.email ?? '—'}</td>
                      <td className="px-4 py-3 font-mono text-gray-700 text-xs">{user.user_id}</td>
                      <td className="px-4 py-3">
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() => revokeMutation.mutate(user.user_id)}
                          disabled={
                            revokeMutation.isPending &&
                            revokeMutation.variables === user.user_id
                          }
                          aria-label={`Revoke ${user.email ?? user.user_id}`}
                        >
                          <UserX className="h-3.5 w-3.5 mr-1" />Revoke
                        </Button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
