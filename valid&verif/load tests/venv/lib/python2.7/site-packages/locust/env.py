class LocustEnviron(object):
    def __init__(self):
        self.request_success = EventHook()
        self.request_failure = EventHook()
        self.locust_error = EventHook()
        
        self.stats = RequestStats(self)
        self.web_app = init_web_app(self)
    
    def run_cli():
        # parse stuff
        
        if options.master:
            runner = MasterLocustRunner(self, locust_classes, options.hatch_rate, options.num_clients, num_requests=options.num_requests, host=options.host, master_host=options.master_host)
        
        self.runner = runner
        
        self.greenlet = gevent.spawn(self._run_web_app)
        greenlet.join(self.greenlet)
    
    def _run_web_app():
        wsgi.WSGIServer(('', port), app, log=None).serve_forever()

