export function AccessPending() {
  return (
    <div className="flex flex-col items-center justify-center h-screen gap-3 text-gray-600">
      <h1 className="text-2xl font-semibold">Access Pending</h1>
      <p className="text-sm">Your account has not been approved yet. Contact an administrator.</p>
      <button
        onClick={() => window.location.reload()}
        className="mt-2 px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded"
      >
        Refresh
      </button>
    </div>
  )
}
