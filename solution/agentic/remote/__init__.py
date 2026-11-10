"""HTTP clients for the UDA-Hub microservices (service/).

The data functions in agentic/tools are decorated with `@remote("<service>")`. When that service's URL is
configured (e.g. CULTPASS_SERVICE_URL), the call goes to the module of the same name here, which sends it
to the service over HTTP. Otherwise the function runs locally against the database, as in the notebooks.
Callers never see the difference: the remote versions take the same arguments and return the same data.
"""
