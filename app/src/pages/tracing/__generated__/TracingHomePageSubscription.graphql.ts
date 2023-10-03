/**
 * @generated SignedSource<<b78a5ee2736aec20d00514e45d08a307>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, GraphQLSubscription } from 'relay-runtime';
export type TracingHomePageSubscription$variables = {};
export type TracingHomePageSubscription$data = {
  readonly traceCount: number;
};
export type TracingHomePageSubscription = {
  response: TracingHomePageSubscription$data;
  variables: TracingHomePageSubscription$variables;
};

const node: ConcreteRequest = (function(){
var v0 = [
  {
    "alias": null,
    "args": null,
    "kind": "ScalarField",
    "name": "traceCount",
    "storageKey": null
  }
];
return {
  "fragment": {
    "argumentDefinitions": [],
    "kind": "Fragment",
    "metadata": null,
    "name": "TracingHomePageSubscription",
    "selections": (v0/*: any*/),
    "type": "Subscription",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": [],
    "kind": "Operation",
    "name": "TracingHomePageSubscription",
    "selections": (v0/*: any*/)
  },
  "params": {
    "cacheID": "4171b7c633238d0a815f10a5a926e1b0",
    "id": null,
    "metadata": {},
    "name": "TracingHomePageSubscription",
    "operationKind": "subscription",
    "text": "subscription TracingHomePageSubscription {\n  traceCount\n}\n"
  }
};
})();

(node as any).hash = "8e52a01c03861770e3ce7cc25bc9d779";

export default node;
