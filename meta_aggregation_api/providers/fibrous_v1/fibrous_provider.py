import asyncio
import ssl
from pathlib import Path
from typing import Dict, Optional, Union

import aiohttp
import ujson
from aiohttp import ClientResponse, ClientResponseError, ServerDisconnectedError
from pydantic import ValidationError
from yarl import URL

from meta_aggregation_api.clients.apm_client import ApmClient
from meta_aggregation_api.config import Config
from meta_aggregation_api.models.meta_agg_models import (
    ProviderPriceResponse,
    ProviderQuoteResponse,
)
from meta_aggregation_api.models.provider_response_models import SwapSources
from meta_aggregation_api.providers.base_provider import BaseProvider
from meta_aggregation_api.utils.errors import (
    AggregationProviderError,
    BaseAggregationProviderError,
    InsufficientLiquidityError,
    EstimationError,
)
from meta_aggregation_api.utils.logger import get_logger

logger = get_logger(__name__)

# Chain ID to Fibrous API URL mapping
CHAIN_ID_TO_API_BASE = {
    8453: 'base',      # Base
    534352: 'scroll',  # Scroll
    999: 'hyperevm',   # HyperEVM
}

# Fibrous error patterns
FIBROUS_ERRORS = {
    'insufficient liquidity': InsufficientLiquidityError,
    'insufficient balance': InsufficientLiquidityError,
    'cannot estimate': EstimationError,
    'no route found': InsufficientLiquidityError,
}


class FibrousProviderV1(BaseProvider):
    """
    Provider for Fibrous Finance DEX Aggregator.
    Docs: https://docs.fibrous.finance/api-reference
    
    Supported Networks:
    - Base (Chain ID: 8453)
    - Scroll (Chain ID: 534352)
    - HyperEVM (Chain ID: 999)
    
    Endpoints:
    - GET /route: Get best route for token swap
    - GET /calldata: Get transaction calldata for swap execution
    - GET /healthCheck: Check API health status
    """
    
    # API Base URLs for each network
    FIBROUS_API_BASE = 'https://api.fibrous.finance/base'
    FIBROUS_API_SCROLL = 'https://api.fibrous.finance/scroll'
    FIBROUS_API_HYPEREVM = 'https://api.fibrous.finance/hyperevm'
    
    with open(Path(__file__).parent / 'config.json') as f:
        PROVIDER_NAME = ujson.load(f)['name']
    
    def __init__(
        self,
        *,
        config: Config,
        session: aiohttp.ClientSession,
        apm_client: ApmClient,
        **_,
    ) -> None:
        super().__init__(config=config, session=session, apm_client=apm_client)
    
    def _get_api_base_url(self, chain_id: int) -> str:
        """
        Get the Fibrous API base URL for a given chain ID.
        
        Args:
            chain_id: The blockchain chain ID
            
        Returns:
            Base URL for the Fibrous API
            
        Raises:
            ValueError: If chain_id is not supported
        """
        chain_name = CHAIN_ID_TO_API_BASE.get(chain_id)
        if not chain_name:
            raise ValueError(f'Fibrous does not support chain_id {chain_id}')
        
        if chain_id == 8453:
            return self.FIBROUS_API_BASE
        elif chain_id == 534352:
            return self.FIBROUS_API_SCROLL
        elif chain_id == 999:
            return self.FIBROUS_API_HYPEREVM
        
        raise ValueError(f'Fibrous does not support chain_id {chain_id}')
    
    async def _make_request(
        self,
        url: URL,
        params: Optional[Dict] = None,
        method: str = 'GET',
        body: Optional[Dict] = None,
    ) -> Union[Dict, list]:
        """
        Make an HTTP request to Fibrous API with error handling.
        
        Args:
            url: Request URL
            params: Query parameters
            method: HTTP method (GET or POST)
            body: Request body for POST requests
            
        Returns:
            Response data as dictionary or list
            
        Raises:
            ClientResponseError: If request fails
        """
        request_function = getattr(self.aiohttp_session, method.lower())
        
        async with request_function(
            str(url),
            params=params,
            timeout=self.REQUEST_TIMEOUT,
            ssl=ssl.SSLContext(),
            json=body,
        ) as response:
            response: ClientResponse
            logger.debug(f'Request {method} {response.url}')
            
            data = await response.read()
            if not data:
                return {}
            
            data = ujson.loads(data)
            
            try:
                response.raise_for_status()
            except ClientResponseError as e:
                # Fix bug with HTTP status code 0
                status = 500 if e.status not in range(100, 600) else e.status
                data['source'] = 'proxied Fibrous API'
                raise ClientResponseError(
                    request_info=e.request_info,
                    history=e.history,
                    status=status,
                    message=[data],
                    headers=e.headers,
                )
        
        return data
    
    async def get_swap_price(
        self,
        buy_token: str,
        sell_token: str,
        sell_amount: int,
        chain_id: int,
        gas_price: Optional[int] = None,
        slippage_percentage: Optional[float] = None,
        taker_address: Optional[str] = None,
        fee_recipient: Optional[str] = None,
        buy_token_percentage_fee: Optional[float] = None,
        **kwargs,
    ) -> ProviderPriceResponse:
        """
        Get swap price from Fibrous /route endpoint.
        
        Args:
            buy_token: Token address to buy
            sell_token: Token address to sell
            sell_amount: Amount to sell in base units
            chain_id: Chain ID
            gas_price: Gas price (optional)
            slippage_percentage: Slippage tolerance (optional)
            taker_address: User address (optional)
            fee_recipient: Fee recipient address (optional)
            buy_token_percentage_fee: Fee percentage (optional)
            
        Returns:
            ProviderPriceResponse with price information
        """
        try:
            api_base = self._get_api_base_url(chain_id)
            url = URL(api_base) / 'route'
            
            # Build query parameters for Fibrous /route endpoint
            query = {
                'tokenInAddress': sell_token,
                'tokenOutAddress': buy_token,
                'amount': str(sell_amount),
            }
            
            if slippage_percentage:
                # Convert slippage to basis points (0.01 = 100 bps)
                query['slippage'] = int(slippage_percentage * 10000)
            
            response = await self._make_request(url, query)
            
        except (
            ClientResponseError,
            asyncio.TimeoutError,
            ServerDisconnectedError,
        ) as e:
            exc = self.handle_exception(
                e, 
                params=query, 
                token_address=sell_token, 
                chain_id=chain_id
            )
            raise exc
        
        # Transform Fibrous response to ProviderPriceResponse
        try:
            # Extract values from Fibrous response
            output_amount = response.get('outputAmount', '0')
            estimated_gas = response.get('estimatedGas', '0')
            
            # Calculate price
            sell_amount_float = float(sell_amount)
            buy_amount_float = float(output_amount)
            price = buy_amount_float / sell_amount_float if sell_amount_float > 0 else 0
            
            # Determine value (native token amount to send)
            value = '0'
            if sell_token.lower() == self.config.NATIVE_TOKEN_ADDRESS:
                value = str(sell_amount)
            
            # Extract route/sources information
            sources = []
            route = response.get('route', [])
            if route:
                # Convert Fibrous route to SwapSources format
                for step in route:
                    protocol = step.get('protocol', 'Unknown')
                    percent = step.get('percent', 100)
                    sources.append(
                        SwapSources(name=protocol, proportion=percent)
                    )
            
            res = ProviderPriceResponse(
                provider=self.PROVIDER_NAME,
                sources=sources,
                buy_amount=str(output_amount),
                gas=str(estimated_gas),
                sell_amount=str(sell_amount),
                gas_price=str(gas_price) if gas_price else '0',
                value=value,
                price=str(price),
                allowance_target=None,
            )
            
        except (KeyError, ValidationError, ValueError) as e:
            exc = self.handle_exception(
                e,
                response=response,
                method='get_swap_price',
                url=str(url),
                params=query,
                chain_id=chain_id,
            )
            raise exc
        
        return res
    
    async def get_swap_quote(
        self,
        buy_token: str,
        sell_token: str,
        sell_amount: int,
        chain_id: int,
        taker_address: str,
        gas_price: Optional[int] = None,
        slippage_percentage: Optional[float] = None,
        fee_recipient: Optional[str] = None,
        buy_token_percentage_fee: Optional[float] = None,
        **kwargs,
    ) -> ProviderQuoteResponse:
        """
        Get swap quote with transaction data from Fibrous /calldata endpoint.
        
        Args:
            buy_token: Token address to buy
            sell_token: Token address to sell
            sell_amount: Amount to sell in base units
            chain_id: Chain ID
            taker_address: User address (required)
            gas_price: Gas price (optional)
            slippage_percentage: Slippage tolerance (optional)
            fee_recipient: Fee recipient address (optional)
            buy_token_percentage_fee: Fee percentage (optional)
            
        Returns:
            ProviderQuoteResponse with transaction data
        """
        if not taker_address:
            raise ValueError('taker_address is required for Fibrous quote')
        
        try:
            api_base = self._get_api_base_url(chain_id)
            url = URL(api_base) / 'calldata'
            
            # Build query parameters for Fibrous /calldata endpoint
            query = {
                'tokenInAddress': sell_token,
                'tokenOutAddress': buy_token,
                'amount': str(sell_amount),
                'userAddress': taker_address,
            }
            
            if slippage_percentage:
                # Convert slippage to basis points (0.01 = 100 bps)
                query['slippage'] = int(slippage_percentage * 10000)
            
            response = await self._make_request(url, query)
            
        except (
            ClientResponseError,
            asyncio.TimeoutError,
            ServerDisconnectedError,
        ) as e:
            exc = self.handle_exception(
                e,
                params=query,
                token_address=sell_token,
                chain_id=chain_id,
                wallet=taker_address,
            )
            raise exc
        
        # Transform Fibrous response to ProviderQuoteResponse
        try:
            # Extract transaction data from Fibrous response
            calldata = response.get('calldata', '0x')
            to_address = response.get('to', '')
            value = response.get('value', '0')
            output_amount = response.get('outputAmount', '0')
            estimated_gas = response.get('estimatedGas', '0')
            
            # Calculate price
            sell_amount_float = float(sell_amount)
            buy_amount_float = float(output_amount)
            price = buy_amount_float / sell_amount_float if sell_amount_float > 0 else 0
            
            # Extract route/sources information
            sources = []
            route = response.get('route', [])
            if route:
                for step in route:
                    protocol = step.get('protocol', 'Unknown')
                    percent = step.get('percent', 100)
                    sources.append(
                        SwapSources(name=protocol, proportion=percent)
                    )
            
            quote = ProviderQuoteResponse(
                sources=sources,
                buy_amount=str(output_amount),
                gas=str(estimated_gas),
                sell_amount=str(sell_amount),
                to=to_address,
                data=calldata,
                gas_price=str(gas_price) if gas_price else '0',
                value=str(value),
                price=str(price),
            )
            
        except (KeyError, ValidationError, ValueError) as e:
            exc = self.handle_exception(
                e,
                response=response,
                method='get_swap_quote',
                url=str(url),
                params=query,
                chain_id=chain_id,
            )
            raise exc
        
        return quote
    
    def handle_exception(
        self,
        exception: Union[ClientResponseError, KeyError, ValidationError],
        **kwargs,
    ) -> BaseAggregationProviderError:
        """
        Handle exceptions and map Fibrous errors to standard error classes.
        
        Args:
            exception: The exception to handle
            **kwargs: Additional context for error logging
            
        Returns:
            BaseAggregationProviderError: Mapped error
        """
        # Try base exception handling first
        exc = super().handle_exception(exception, **kwargs)
        if exc:
            logger.error(*exc.to_log_args(), extra=exc.to_dict())
            return exc
        
        # Handle Fibrous-specific errors
        msg = str(exception)
        if isinstance(exception, ClientResponseError):
            if isinstance(exception.message, list) and isinstance(
                exception.message[0], dict
            ):
                msg = exception.message[0].get(
                    'error',
                    exception.message[0].get('message', str(exception))
                )
        
        # Map Fibrous errors to standard error classes
        error_class = AggregationProviderError
        for error_pattern, error_cls in FIBROUS_ERRORS.items():
            if error_pattern.lower() in msg.lower():
                error_class = error_cls
                break
        
        exc = error_class(
            self.PROVIDER_NAME,
            msg,
            url=kwargs.get('url', ''),
            **kwargs,
        )
        
        logger.warning(*exc.to_log_args(), extra=exc.to_dict())
        return exc
    
    async def health_check(self, chain_id: int) -> bool:
        """
        Check if Fibrous API is healthy for a given chain.
        
        Args:
            chain_id: Chain ID to check
            
        Returns:
            True if healthy, False otherwise
        """
        try:
            api_base = self._get_api_base_url(chain_id)
            url = URL(api_base) / 'healthCheck'
            
            response = await self._make_request(url)
            
            # Check if response indicates health
            status = response.get('status', '')
            return status.lower() == 'ok' or response.get('healthy', False)
            
        except Exception as e:
            logger.error(f'Fibrous health check failed for chain {chain_id}: {e}')
            return False

